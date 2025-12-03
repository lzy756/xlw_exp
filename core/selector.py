"""FAP Selector for v2 fair-weighted aggregation."""

from typing import Dict, List
import numpy as np


class FAPSelector:
    """FAP selector for computing fair factors in v2 aggregation.
    
    In v2, the selector's role is to compute domain aggregation weights,
    not to decide whether to aggregate. θ is aggregated every round.
    """

    def __init__(
        self,
        domains: List[str],
        w1: float = 1.0,
        w2: float = 0.3,
        tau: float = 1.0
    ):
        """Initialize FAP Selector.

        Args:
            domains: List of domain names
            w1: Weight for loss L_e (high loss → high weight)
            w2: Weight for drift Δ_e (high drift → low weight)
            tau: Softmax temperature
        """
        self.domains = domains
        self.w1 = w1
        self.w2 = w2
        self.tau = tau

    def _zscore(self, values: np.ndarray) -> np.ndarray:
        """Z-score normalization.

        Args:
            values: Input array

        Returns:
            Normalized array
        """
        if len(values) == 0:
            return values

        mean = np.mean(values)
        std = np.std(values)

        if std < 1e-9:
            return np.zeros_like(values)

        return (values - mean) / std

    def compute_fair_factors(
        self,
        L_map: Dict[str, float],
        drift_map: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute fairness factors for weighted aggregation.
        
        Formula: q_e = softmax( w1 * L̃_e - w2 * Δ̃_e, τ )
        
        Where:
        - L̃_e: Z-score normalized EMA validation loss
        - Δ̃_e: Z-score normalized prototype drift
        - w1, w2: Weight coefficients
        - τ: Softmax temperature
        
        Higher loss → higher weight (domains needing more attention)
        Higher drift → lower weight (unstable contributions may be harmful)

        Args:
            L_map: Domain -> EMA loss
            drift_map: Domain -> drift score Δ_e

        Returns:
            fair_factors: {domain: factor} normalized fairness factors
        """
        # Extract values in consistent order
        L_values = np.array([L_map.get(d, 0.0) for d in self.domains], dtype=np.float64)
        drift_values = np.array([drift_map.get(d, 0.0) for d in self.domains], dtype=np.float64)
        
        # Z-score normalization
        L_norm = self._zscore(L_values)
        drift_norm = self._zscore(drift_values)
        
        # Compute scores: high loss adds, high drift subtracts
        scores = self.w1 * L_norm - self.w2 * drift_norm
        
        # Softmax to get probability distribution
        scores = scores / self.tau
        scores = scores - scores.max()  # Numerical stability
        exp_scores = np.exp(scores)
        q = exp_scores / exp_scores.sum()
        
        return {d: q[i] for i, d in enumerate(self.domains)}

    def get_uniform_factors(self) -> Dict[str, float]:
        """Get uniform fairness factors (for standard FedAvg).

        Returns:
            Uniform factors: {domain: 1/n}
        """
        n = len(self.domains)
        return {d: 1.0 / n for d in self.domains}
