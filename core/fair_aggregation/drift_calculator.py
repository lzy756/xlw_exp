"""Drift calculator for FAP v3 using S1 statistic aggregation.

This module implements section A3: Drift computation using:
- Random projection for dimensionality reduction
- S1 statistic aggregation (per-class sum and count)
- Class prototype computation
- Drift score calculation with snapshot mechanism

Reference Formulas:
    z_tilde = R^T @ z                           # Random projection
    mu_tilde_{e,c} = S_{e,c} / (N_{e,c} + eps)  # Class prototype
    Delta_e = sum_c pi_{e,c} * ||mu_{e,c}(t) - mu_{e,c}(t-tau_d)||_2

Where:
    - R: Fixed random projection matrix (d x d')
    - S_{e,c}: Aggregated projected feature sum for class c
    - N_{e,c}: Aggregated sample count for class c
    - pi_{e,c}: Class frequency weight
    - tau_d: Snapshot interval (default: 5 rounds)
"""

import math
from typing import Dict, List, Optional, Tuple
import torch


class ClientStatistics:
    """Container for client-level S1 statistics.

    Each client computes and uploads:
        - s_tilde_{k,c}: Projected feature sum per class
        - n_{k,c}: Sample count per class
    """

    def __init__(
        self,
        class_sums: torch.Tensor,
        class_counts: torch.Tensor
    ):
        """Initialize client statistics.

        Args:
            class_sums: Tensor of shape [num_classes, d'] - projected feature sums
            class_counts: Tensor of shape [num_classes] - sample counts per class
        """
        self.class_sums = class_sums
        self.class_counts = class_counts


class DriftCalculator:
    """Computes prototype drift using S1 statistic aggregation.

    This class implements section A3 of the FAP v3 algorithm:
    1. Maintains fixed random projection matrix R
    2. Aggregates client statistics to domain-level prototypes
    3. Computes drift by comparing with historical snapshots

    The drift score reflects distribution shift within a domain,
    used to penalize unstable domains in cloud aggregation.
    """

    def __init__(
        self,
        feature_dim: int,
        proj_dim: int = 64,
        num_classes: int = 126,
        tau_d: int = 5,
        seed: int = 42,
        device: str = 'cuda'
    ):
        """Initialize drift calculator.

        Args:
            feature_dim: Original feature dimension d (e.g., 512 for ResNet)
            proj_dim: Projected dimension d' (default: 64)
            num_classes: Number of classes C
            tau_d: Snapshot interval in rounds (default: 5)
            seed: Random seed for projection matrix (must be consistent)
            device: Compute device
        """
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.tau_d = tau_d
        self.device = device

        # Generate fixed random projection matrix R in R^{d x d'}
        # Using Gaussian random projection (JL-lemma guarantees)
        torch.manual_seed(seed)
        self.R = torch.randn(
            feature_dim, proj_dim, device=device
        ) / math.sqrt(feature_dim)

        # Current round counter
        self._round = 0

        # Domain-level aggregated statistics
        self._domain_sums: Dict[str, torch.Tensor] = {}
        self._domain_counts: Dict[str, torch.Tensor] = {}

        # Current prototypes (computed from aggregated stats)
        self._current_proto: Dict[str, torch.Tensor] = {}

        # Snapshot prototypes (for drift computation)
        self._snapshot_proto: Dict[str, Optional[torch.Tensor]] = {}
        self._snapshot_round: Dict[str, int] = {}

        # Cached drift scores
        self._drift_cache: Dict[str, float] = {}

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply random projection to features (A3.3).

        Formula:
            z_tilde = R^T @ z

        Args:
            features: Feature tensor of shape [N, d]

        Returns:
            Projected features of shape [N, d']
        """
        # Ensure consistent dtype
        features = features.to(dtype=self.R.dtype, device=self.device)
        return features @ self.R

    def compute_client_statistics(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> ClientStatistics:
        """Compute S1 statistics for a client (A3.2).

        For each class c, computes:
            s_tilde_{k,c} = sum_{(x,y)=c} z_tilde
            n_{k,c} = count of samples with label c

        Args:
            features: Raw features [N, d] from backbone h_theta
            labels: Class labels [N]

        Returns:
            ClientStatistics containing per-class sums and counts
        """
        # Project features
        projected = self.project_features(features)

        # Initialize accumulators
        class_sums = torch.zeros(
            self.num_classes, self.proj_dim,
            dtype=projected.dtype, device=self.device
        )
        class_counts = torch.zeros(
            self.num_classes,
            dtype=torch.float32, device=self.device
        )

        # Aggregate per class
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                class_sums[c] = projected[mask].sum(dim=0)
                class_counts[c] = mask.sum().float()

        return ClientStatistics(class_sums, class_counts)

    def aggregate_domain_statistics(
        self,
        domain: str,
        client_stats: List[ClientStatistics]
    ) -> None:
        """Aggregate client statistics to domain level (A3.4).

        Formula:
            S_tilde_{e,c} = sum_{k in K_e} s_tilde_{k,c}
            N_{e,c} = sum_{k in K_e} n_{k,c}

        Args:
            domain: Domain name
            client_stats: List of ClientStatistics from domain clients
        """
        if not client_stats:
            return

        # Sum all client statistics
        total_sums = torch.zeros(
            self.num_classes, self.proj_dim,
            dtype=client_stats[0].class_sums.dtype,
            device=self.device
        )
        total_counts = torch.zeros(
            self.num_classes,
            dtype=torch.float32, device=self.device
        )

        for stats in client_stats:
            total_sums += stats.class_sums.to(self.device)
            total_counts += stats.class_counts.to(self.device)

        self._domain_sums[domain] = total_sums
        self._domain_counts[domain] = total_counts

    def compute_prototypes(self, domain: str) -> torch.Tensor:
        """Compute domain-level class prototypes (A3.4).

        Formula:
            mu_tilde_{e,c} = S_tilde_{e,c} / (N_{e,c} + epsilon)

        Args:
            domain: Domain name

        Returns:
            Prototype tensor of shape [num_classes, d']
        """
        eps = 1e-6

        sums = self._domain_sums.get(domain)
        counts = self._domain_counts.get(domain)

        if sums is None or counts is None:
            # No statistics available, return zeros
            return torch.zeros(
                self.num_classes, self.proj_dim,
                device=self.device
            )

        # Compute prototypes with epsilon for numerical stability
        prototypes = sums / (counts.unsqueeze(1) + eps)
        self._current_proto[domain] = prototypes

        return prototypes

    def update_snapshot(self, domain: str) -> bool:
        """Update prototype snapshot if tau_d rounds have passed (A3.5).

        Snapshots are taken every tau_d rounds and used as reference
        for drift computation.

        Args:
            domain: Domain name

        Returns:
            True if snapshot was updated, False otherwise
        """
        current_round = self._round

        # Check if it's time to update snapshot
        last_snapshot = self._snapshot_round.get(domain, -self.tau_d)
        if current_round - last_snapshot >= self.tau_d:
            # Take snapshot of current prototypes
            if domain in self._current_proto:
                self._snapshot_proto[domain] = self._current_proto[domain].clone()
                self._snapshot_round[domain] = current_round
                return True

        return False

    def compute_drift(self, domain: str) -> float:
        """Compute prototype drift score for a domain (A3.5).

        Formula:
            Delta_e = sum_c pi_{e,c} * ||mu_{e,c}(t) - mu_{e,c}(t-tau_d)||_2

        Where:
            pi_{e,c} = N_{e,c} / (sum_c' N_{e,c'} + epsilon)

        Args:
            domain: Domain name

        Returns:
            Drift score Delta_e (0 if no snapshot available)
        """
        eps = 1e-6

        # Get current prototypes
        current = self._current_proto.get(domain)
        snapshot = self._snapshot_proto.get(domain)

        if current is None or snapshot is None:
            # No comparison possible
            drift = 0.0
            self._drift_cache[domain] = drift
            return drift

        # Get class counts for frequency weights
        counts = self._domain_counts.get(domain)
        if counts is None:
            drift = 0.0
            self._drift_cache[domain] = drift
            return drift

        # Compute class frequency weights pi_{e,c}
        total_count = counts.sum() + eps
        pi = counts / total_count  # [num_classes]

        # Compute per-class L2 distance
        diff = current - snapshot  # [num_classes, d']
        l2_norms = torch.norm(diff, p=2, dim=1)  # [num_classes]

        # Skip classes with zero count (avoid spurious drift)
        valid_mask = counts > 0
        if not valid_mask.any():
            drift = 0.0
        else:
            # Weighted sum of L2 distances
            drift = (pi * l2_norms * valid_mask.float()).sum().item()

        self._drift_cache[domain] = drift
        return drift

    def step_round(self) -> None:
        """Advance to next round."""
        self._round += 1

    def get_cached_drift(self, domain: str) -> float:
        """Get cached drift score for a domain.

        Args:
            domain: Domain name

        Returns:
            Cached drift score (0 if not computed)
        """
        return self._drift_cache.get(domain, 0.0)

    def get_all_drifts(self, domains: List[str]) -> Dict[str, float]:
        """Get drift scores for all domains.

        Args:
            domains: List of domain names

        Returns:
            Dictionary mapping domain -> drift score
        """
        return {d: self._drift_cache.get(d, 0.0) for d in domains}

    def reset_domain_accumulators(self, domain: str) -> None:
        """Reset domain statistics accumulators for next round.

        Args:
            domain: Domain name
        """
        if domain in self._domain_sums:
            self._domain_sums[domain].zero_()
        if domain in self._domain_counts:
            self._domain_counts[domain].zero_()

    def get_projection_matrix(self) -> torch.Tensor:
        """Get the random projection matrix R.

        Clients need this matrix to project features before
        computing statistics.

        Returns:
            Projection matrix R of shape [d, d']
        """
        return self.R
