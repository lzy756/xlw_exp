"""Edge manager for domain-specific state management."""

import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from utils.common import EMA


class EdgeManager:
    """Manages domain-specific states and metrics for edge servers."""

    def __init__(
        self,
        model: nn.Module,
        domains: List[str],
        num_classes: int = 126,
        proj_dim: int = 64,
        device: str = 'cuda'
    ):
        """Initialize EdgeManager.

        Args:
            model: Neural network model
            domains: List of domain names
            num_classes: Number of classes
            proj_dim: Dimension for random projection (prototypes)
            device: Device for tensors
        """
        self.model = model
        self.domains = domains
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.device = device

        # Domain-specific parameter storage
        self.domain_phi: Dict[str, Optional[Dict[str, torch.Tensor]]] = {
            d: None for d in domains
        }

        # EMA tracking for losses
        self.L_ema: Dict[str, EMA] = {
            d: EMA(beta=0.9) for d in domains
        }

        # Selection metrics
        self.coverage: Dict[str, float] = {d: 0.0 for d in domains}
        self.stay_bonus: Dict[str, float] = {d: 0.0 for d in domains}
        self.last_aggregator: Optional[str] = None

        # Prototype tracking
        # Fixed random projection matrix (deterministic with seed=0)
        torch.manual_seed(0)
        self.proj = torch.randn(512, proj_dim, device=device) / math.sqrt(512)

        # Prototype accumulators
        self.proto_sum: Dict[str, torch.Tensor] = {
            d: torch.zeros(num_classes, proj_dim, device=device)
            for d in domains
        }
        self.proto_cnt: Dict[str, torch.Tensor] = {
            d: torch.zeros(num_classes, device=device)
            for d in domains
        }

        # Previous prototypes for drift computation
        self.prev_proto: Dict[str, Optional[torch.Tensor]] = {
            d: None for d in domains
        }

    def get_phi(self, domain: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get domain-specific parameters.

        Args:
            domain: Domain name

        Returns:
            State dictionary for domain's phi parameters or None
        """
        return self.domain_phi.get(domain)

    def set_phi(self, domain: str, state_dict: Dict[str, torch.Tensor]) -> None:
        """Set domain-specific parameters.

        Args:
            domain: Domain name
            state_dict: State dictionary containing phi parameters
        """
        self.domain_phi[domain] = state_dict

    def begin_round(self) -> None:
        """Called at the beginning of each round to update coverage counters."""
        for domain in self.domains:
            self.coverage[domain] += 1.0

    def end_round_with_aggregator(self, agg_domain: str) -> None:
        """Called when a domain is selected as aggregator.

        Args:
            agg_domain: Domain selected as aggregator
        """
        # Reset coverage for selected domain
        self.coverage[agg_domain] = 0.0

        # Update stability bonus
        for domain in self.domains:
            if domain == agg_domain and self.last_aggregator == agg_domain:
                self.stay_bonus[domain] = 0.2  # Stability bonus
            else:
                self.stay_bonus[domain] = 0.0

        self.last_aggregator = agg_domain

    def update_eval_stats(
        self,
        domain: str,
        val_loss: float,
        feats: torch.Tensor,
        labels: torch.Tensor,
        tau: float = 1.0
    ) -> float:
        """Update evaluation statistics and accumulate prototypes.

        Args:
            domain: Domain name
            val_loss: Validation loss
            feats: Feature vectors (N x 512)
            labels: Class labels (N,)
            tau: Temperature parameter (unused here, for compatibility)

        Returns:
            Updated EMA loss L_e
        """
        # Update EMA loss
        L_e = self.L_ema[domain].update(val_loss)

        # Project features to lower dimension
        with torch.no_grad():
            projected_feats = feats @ self.proj  # N x proj_dim

            # Accumulate per-class prototypes
            for class_id in range(self.num_classes):
                class_mask = labels == class_id
                if class_mask.any():
                    class_feats = projected_feats[class_mask]
                    self.proto_sum[domain][class_id] += class_feats.sum(dim=0)
                    self.proto_cnt[domain][class_id] += class_mask.sum().float()

        return L_e

    def compute_drift(self, domain: str) -> float:
        """Compute prototype drift for a domain.

        Args:
            domain: Domain name

        Returns:
            Drift score Δ_e
        """
        with torch.no_grad():
            # Compute current prototypes
            counts = self.proto_cnt[domain].unsqueeze(1) + 1e-9
            cur_proto = self.proto_sum[domain] / counts  # [num_classes, proj_dim]

            if self.prev_proto[domain] is None:
                # First round, no drift
                drift = 0.0
            else:
                # Compute weighted L2 distance
                prev_proto = self.prev_proto[domain]

                # Class frequency weights (pi)
                total_count = self.proto_cnt[domain].sum() + 1e-9
                pi = self.proto_cnt[domain] / total_count  # [num_classes]

                # L2 distance per class
                diff_norms = (cur_proto - prev_proto).norm(p=2, dim=1)  # [num_classes]

                # Weighted drift
                drift = (pi * diff_norms).sum().item()

            # Update previous prototypes
            self.prev_proto[domain] = cur_proto.detach().clone()

            # Reset accumulators
            self.proto_sum[domain].zero_()
            self.proto_cnt[domain].zero_()

            return drift

    def get_metrics_for_selection(self) -> Dict[str, Dict[str, float]]:
        """Get all metrics needed for aggregation point selection.

        Returns:
            Dictionary with L_map, drift_map, cover_map, stay_map
        """
        L_map = {}
        drift_map = {}
        cover_map = {}
        stay_map = {}

        for domain in self.domains:
            # Get EMA loss
            L_e = self.L_ema[domain].get()
            L_map[domain] = L_e if L_e is not None else 0.0

            # Compute drift (this also resets accumulators)
            drift_map[domain] = self.compute_drift(domain)

            # Coverage and stability
            cover_map[domain] = self.coverage[domain]
            stay_map[domain] = self.stay_bonus[domain]

        return {
            'L_map': L_map,
            'drift_map': drift_map,
            'cover_map': cover_map,
            'stay_map': stay_map
        }