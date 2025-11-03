"""FAP-Float(S) selector for aggregation point selection."""

from typing import Dict, List, Tuple
import numpy as np


class FAPFloatSoftmax:
    """Fairness-Aware floating aggregation Point selector with Softmax sampling."""

    def __init__(
        self,
        domains: List[str],
        w1: float = 1.0,
        w2: float = 0.5,
        w3: float = 0.3,
        w4: float = 0.2,
        tau: float = 1.0
    ):
        """Initialize FAP-Float(S) selector.

        Args:
            domains: List of domain names
            w1: Weight for loss L_e
            w2: Weight for drift Δ_e
            w3: Weight for coverage H_e
            w4: Weight for stability S_e^stay
            tau: Softmax temperature
        """
        self.domains = domains
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.tau = tau

    def _norm(self, xs: np.ndarray) -> np.ndarray:
        """Z-score normalization.

        Args:
            xs: Input array

        Returns:
            Normalized array
        """
        if len(xs) == 0:
            return xs

        mean = np.mean(xs)
        std = np.std(xs)

        if std < 1e-9:
            # All values are the same
            return np.zeros_like(xs)

        return (xs - mean) / std

    def select(
        self,
        L_map: Dict[str, float],
        drift_map: Dict[str, float],
        cover_map: Dict[str, float],
        stay_map: Dict[str, float]
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """Select aggregation point using FAP-Float(S) scoring.

        Args:
            L_map: Domain -> EMA loss
            drift_map: Domain -> drift score Δ_e
            cover_map: Domain -> coverage count H_e
            stay_map: Domain -> stability bonus S_e^stay

        Returns:
            Tuple of (selected_domain, probabilities, raw_scores)
        """
        # Validate input
        if not all(domain in L_map for domain in self.domains):
            raise ValueError("L_map missing some domains")
        if not all(domain in drift_map for domain in self.domains):
            raise ValueError("drift_map missing some domains")
        if not all(domain in cover_map for domain in self.domains):
            raise ValueError("cover_map missing some domains")
        if not all(domain in stay_map for domain in self.domains):
            raise ValueError("stay_map missing some domains")

        # Extract values in consistent order
        L_values = np.array([L_map[d] for d in self.domains])
        drift_values = np.array([drift_map[d] for d in self.domains])
        cover_values = np.array([cover_map[d] for d in self.domains])
        stay_values = np.array([stay_map[d] for d in self.domains])

        # Validate values are finite and non-negative
        if not np.all(np.isfinite(L_values)) or np.any(L_values < 0):
            raise ValueError("L_values must be finite and non-negative")
        if not np.all(np.isfinite(drift_values)) or np.any(drift_values < 0):
            raise ValueError("drift_values must be finite and non-negative")
        if not np.all(np.isfinite(cover_values)) or np.any(cover_values < 0):
            raise ValueError("cover_values must be finite and non-negative")
        if not np.all(np.isfinite(stay_values)) or np.any(stay_values < 0):
            raise ValueError("stay_values must be finite and non-negative")

        # Normalize each metric
        L_norm = self._norm(L_values)
        drift_norm = self._norm(drift_values)
        cover_norm = self._norm(cover_values)
        stay_norm = stay_values  # No normalization for binary bonus

        # Compute weighted scores
        scores = (
            self.w1 * L_norm +
            self.w2 * drift_norm +
            self.w3 * cover_norm +
            self.w4 * stay_norm
        )

        # Handle edge case: all scores identical
        if np.std(scores) < 1e-9:
            # Return uniform probabilities
            probabilities = np.ones(len(self.domains)) / len(self.domains)
            selected_idx = np.random.choice(len(self.domains))
        else:
            # Softmax with temperature
            exp_scores = np.exp(scores / self.tau)
            probabilities = exp_scores / exp_scores.sum()

            # Sample from probabilities
            selected_idx = np.random.choice(
                len(self.domains),
                p=probabilities
            )

        selected_domain = self.domains[selected_idx]

        return selected_domain, probabilities, scores