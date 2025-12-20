"""Cloud-side weighted aggregation for FAP v3.

This module implements sections B2-B5: Cloud aggregation with:
- Z-score normalization of domain states
- Score computation with loss compensation and drift penalty
- Sample-weight fusion with softmax domain weights
- Clipping, normalization, and EMA smoothing

Reference Formulas:
    B2: L_tilde_e = (L_bar_e - mu_L) / (sigma_L + eps)  # Z-score
    B3: score_e = w1 * L_tilde_e - w2 * Delta_tilde_e + w3 * H_e
        q_e = softmax(score_e / tau)
    B4: alpha_raw_e = (b_e * q_e) / sum(b_j * q_j)
        alpha_e = clip(alpha_raw_e, floor, ceil)  # then renormalize
    B5: theta_{t+1} = theta_t + sum_e alpha_e * Delta_theta_e
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class CloudAggregator:
    """Computes fair aggregation weights for cloud-side theta update.

    This class implements the cloud-side logic for FAP v3:
    1. Receives domain states from DC/Edge servers
    2. Normalizes and scores each domain
    3. Computes final aggregation weights with safety bounds
    4. Optionally applies EMA smoothing for stability

    The key insight is that domains with:
    - Higher loss → Higher weight (need more attention)
    - Higher drift → Lower weight (unstable, may be harmful)
    - Higher coverage gap → Higher weight (long-term fairness)
    """

    def __init__(
        self,
        domains: List[str],
        w1: float = 1.0,
        w2: float = 0.3,
        w3: float = 0.5,
        tau: float = 1.0,
        gamma: float = 0.5,
        alpha_floor: float = 0.05,
        alpha_ceil: float = 0.7,
        ema_beta: float = 0.8,
        zscore_clip: float = 3.0
    ):
        """Initialize cloud aggregator.

        Args:
            domains: List of domain names
            w1: Weight for loss compensation (higher loss → higher weight)
            w2: Weight for drift penalty (higher drift → lower weight)
            w3: Weight for coverage compensation H_e
            tau: Softmax temperature (lower → sharper distribution)
            gamma: Sample weight exponent (0=ignore samples, 1=full FedAvg)
            alpha_floor: Minimum aggregation weight per domain
            alpha_ceil: Maximum aggregation weight per domain
            ema_beta: EMA smoothing factor for final weights (0=no smoothing)
            zscore_clip: Clip z-scores to [-clip, clip] for robustness
        """
        self.domains = domains
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.tau = tau
        self.gamma = gamma
        self.alpha_floor = alpha_floor
        self.alpha_ceil = alpha_ceil
        self.ema_beta = ema_beta
        self.zscore_clip = zscore_clip

        # Previous weights for EMA smoothing
        self._prev_weights: Optional[Dict[str, float]] = None

    def _zscore(self, values: np.ndarray, clip: Optional[float] = None) -> np.ndarray:
        """Z-score normalization with optional clipping (B2).

        Formula:
            z = (x - mean) / (std + eps)
            z_clipped = clip(z, -clip, clip)

        Args:
            values: Input array
            clip: Optional clipping bound

        Returns:
            Normalized (and optionally clipped) array
        """
        eps = 1e-6

        if len(values) == 0:
            return values

        mean = np.mean(values)
        std = np.std(values)

        if std < eps:
            # All values are the same, return zeros
            return np.zeros_like(values)

        z = (values - mean) / (std + eps)

        if clip is not None:
            z = np.clip(z, -clip, clip)

        return z

    def _softmax(self, scores: np.ndarray, tau: float = 1.0) -> np.ndarray:
        """Numerically stable softmax.

        Args:
            scores: Input scores
            tau: Temperature parameter

        Returns:
            Probability distribution
        """
        scaled = scores / tau
        scaled = scaled - np.max(scaled)  # Numerical stability
        exp_scores = np.exp(scaled)
        return exp_scores / np.sum(exp_scores)

    def compute_scores(
        self,
        L_map: Dict[str, float],
        drift_map: Dict[str, float],
        H_map: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute domain scores and softmax weights (B2-B3).

        Formula:
            L_tilde_e = zscore(L_bar_e)
            Delta_tilde_e = zscore(Delta_e)
            score_e = w1 * L_tilde_e - w2 * Delta_tilde_e + w3 * H_e
            q_e = softmax(score_e / tau)

        Args:
            L_map: {domain: EMA_loss}
            drift_map: {domain: drift_score}
            H_map: {domain: coverage_compensation}

        Returns:
            Tuple of (raw_scores, softmax_weights) as dicts
        """
        # Extract values in consistent domain order
        L_values = np.array([L_map.get(d, 0.0) for d in self.domains], dtype=np.float64)
        drift_values = np.array([drift_map.get(d, 0.0) for d in self.domains], dtype=np.float64)
        H_values = np.array([H_map.get(d, 0.0) for d in self.domains], dtype=np.float64)

        # Z-score normalization with clipping (B2)
        L_norm = self._zscore(L_values, clip=self.zscore_clip)
        drift_norm = self._zscore(drift_values, clip=self.zscore_clip)

        # Compute scores (B3)
        # Higher loss → higher score (compensation)
        # Higher drift → lower score (penalty)
        # Higher H → higher score (fairness compensation)
        scores = self.w1 * L_norm - self.w2 * drift_norm + self.w3 * H_values

        # Softmax to get state-based weights
        q = self._softmax(scores, self.tau)

        # Convert to dicts
        score_map = {d: scores[i] for i, d in enumerate(self.domains)}
        q_map = {d: q[i] for i, d in enumerate(self.domains)}

        return score_map, q_map

    def compute_sample_weights(
        self,
        n_map: Dict[str, int]
    ) -> Dict[str, float]:
        """Compute sample-based weights with gamma exponent (B4 part 1).

        Formula:
            b_e = (n_e / sum_j n_j) ^ gamma

        Args:
            n_map: {domain: sample_count}

        Returns:
            Sample weights {domain: b_e}
        """
        eps = 1e-9

        # Extract sample counts
        n_values = np.array([n_map.get(d, 0) for d in self.domains], dtype=np.float64)
        total_n = np.sum(n_values) + eps

        # Normalize and apply gamma exponent
        normalized = n_values / total_n
        b = np.power(normalized + eps, self.gamma)

        return {d: b[i] for i, d in enumerate(self.domains)}

    def compute_aggregation_weights(
        self,
        L_map: Dict[str, float],
        drift_map: Dict[str, float],
        H_map: Dict[str, float],
        n_map: Dict[str, int]
    ) -> Dict[str, float]:
        """Compute final aggregation weights with all safety measures (B4).

        Formula:
            alpha_raw_e = (b_e * q_e) / sum_j(b_j * q_j)
            alpha_clip_e = clip(alpha_raw_e, alpha_floor, alpha_ceil)
            alpha_e = alpha_clip_e / sum_j alpha_clip_j  # renormalize
            alpha_e = beta * alpha_prev_e + (1-beta) * alpha_e  # EMA

        Args:
            L_map: {domain: EMA_loss}
            drift_map: {domain: drift_score}
            H_map: {domain: coverage_compensation}
            n_map: {domain: sample_count}

        Returns:
            Final aggregation weights {domain: alpha_e}
        """
        eps = 1e-9

        # Get state-based weights q_e
        _, q_map = self.compute_scores(L_map, drift_map, H_map)

        # Get sample-based weights b_e
        b_map = self.compute_sample_weights(n_map)

        # Combine: alpha_raw = b * q / sum(b * q)
        raw_products = np.array([
            b_map[d] * q_map[d] for d in self.domains
        ], dtype=np.float64)
        alpha_raw = raw_products / (np.sum(raw_products) + eps)

        # Clip to [floor, ceil]
        alpha_clipped = np.clip(alpha_raw, self.alpha_floor, self.alpha_ceil)

        # Renormalize after clipping
        alpha_norm = alpha_clipped / (np.sum(alpha_clipped) + eps)

        # Apply EMA smoothing if previous weights exist
        if self._prev_weights is not None and self.ema_beta > 0:
            prev_array = np.array([
                self._prev_weights.get(d, alpha_norm[i])
                for i, d in enumerate(self.domains)
            ], dtype=np.float64)
            alpha_final = self.ema_beta * prev_array + (1 - self.ema_beta) * alpha_norm
            # Renormalize after EMA
            alpha_final = alpha_final / (np.sum(alpha_final) + eps)
        else:
            alpha_final = alpha_norm

        # Store for next round
        result = {d: alpha_final[i] for i, d in enumerate(self.domains)}
        self._prev_weights = result.copy()

        return result

    def get_uniform_weights(self) -> Dict[str, float]:
        """Get uniform aggregation weights (fallback/baseline).

        Returns:
            Uniform weights {domain: 1/E}
        """
        n = len(self.domains)
        return {d: 1.0 / n for d in self.domains}

    def get_sample_only_weights(self, n_map: Dict[str, int]) -> Dict[str, float]:
        """Get sample-proportional weights only (standard FedAvg).

        Args:
            n_map: {domain: sample_count}

        Returns:
            Sample-proportional weights {domain: n_e / sum n_j}
        """
        eps = 1e-9
        n_values = np.array([n_map.get(d, 0) for d in self.domains], dtype=np.float64)
        total = np.sum(n_values) + eps
        weights = n_values / total
        return {d: weights[i] for i, d in enumerate(self.domains)}

    def reset(self) -> None:
        """Reset internal state (previous weights)."""
        self._prev_weights = None

    def get_debug_info(
        self,
        L_map: Dict[str, float],
        drift_map: Dict[str, float],
        H_map: Dict[str, float],
        n_map: Dict[str, int]
    ) -> Dict[str, Dict[str, float]]:
        """Get detailed debug information for logging/visualization.

        Args:
            L_map, drift_map, H_map, n_map: Domain state maps

        Returns:
            Dictionary with intermediate computation results
        """
        score_map, q_map = self.compute_scores(L_map, drift_map, H_map)
        b_map = self.compute_sample_weights(n_map)
        alpha_map = self.compute_aggregation_weights(L_map, drift_map, H_map, n_map)

        # Z-scored values for debugging
        L_values = np.array([L_map.get(d, 0.0) for d in self.domains])
        drift_values = np.array([drift_map.get(d, 0.0) for d in self.domains])
        L_zscore = self._zscore(L_values, clip=self.zscore_clip)
        drift_zscore = self._zscore(drift_values, clip=self.zscore_clip)

        return {
            'L_raw': L_map,
            'L_zscore': {d: L_zscore[i] for i, d in enumerate(self.domains)},
            'drift_raw': drift_map,
            'drift_zscore': {d: drift_zscore[i] for i, d in enumerate(self.domains)},
            'H': H_map,
            'score': score_map,
            'q_softmax': q_map,
            'b_sample': b_map,
            'alpha_final': alpha_map
        }
