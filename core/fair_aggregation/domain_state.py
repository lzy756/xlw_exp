"""Domain-level state management for FAP v3 fair-weighted aggregation.

This module implements the DC/Edge side state management including:
- A1: Per-domain state storage (L_ema, drift, gap, sample_count)
- A2: Validation loss EMA tracking
- A4: Coverage compensation (H_e) for long-term weight imbalance correction

Reference Formula:
    H_e(t) = tanh(gap_e(t) / T_H)

Where:
    - gap_e: Consecutive rounds where domain weight < alpha_min
    - T_H: Compensation scale parameter (default: 15)
"""

import math
from typing import Dict, List, Optional


class EMATracker:
    """Exponential Moving Average tracker for validation loss.

    Formula:
        L_bar_e(t) = alpha_L * L_bar_e(t-1) + (1 - alpha_L) * L_raw_e(t)
    """

    def __init__(self, alpha: float = 0.9):
        """Initialize EMA tracker.

        Args:
            alpha: Smoothing factor (higher = more smoothing)
        """
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, raw_value: float) -> float:
        """Update EMA with new raw value.

        Args:
            raw_value: New observation L_raw_e(t)

        Returns:
            Updated EMA value L_bar_e(t)
        """
        if self._value is None:
            self._value = raw_value
        else:
            self._value = self.alpha * self._value + (1 - self.alpha) * raw_value
        return self._value

    def get(self) -> Optional[float]:
        """Get current EMA value."""
        return self._value

    def reset(self) -> None:
        """Reset tracker state."""
        self._value = None


class DomainStateManager:
    """Manages per-domain states for FAP v3 aggregation.

    This class tracks domain-level metrics required for fair-weighted
    cloud aggregation, implementing formulas from sections A1, A2, and A4.

    Maintained States (per domain e):
        - L_ema[e]: EMA validation loss (updated each round)
        - drift[e]: Prototype drift score (computed by DriftCalculator)
        - gap[e]: Coverage compensation counter
        - H[e]: Coverage compensation factor = tanh(gap_e / T_H)
        - n[e]: Domain sample count
    """

    def __init__(
        self,
        domains: List[str],
        alpha_L: float = 0.9,
        alpha_min: float = 0.1,
        T_H: float = 15.0
    ):
        """Initialize domain state manager.

        Args:
            domains: List of domain names
            alpha_L: EMA smoothing factor for validation loss
            alpha_min: Threshold for triggering coverage compensation
                       (suggested: 0.5/E or 1/E where E = num_domains)
            T_H: Scale parameter for tanh compensation (suggested: 10~20)
        """
        self.domains = domains
        self.alpha_L = alpha_L
        self.alpha_min = alpha_min
        self.T_H = T_H

        # A2: EMA loss trackers
        self.L_ema: Dict[str, EMATracker] = {
            d: EMATracker(alpha=alpha_L) for d in domains
        }

        # A3: Drift scores (set externally by DriftCalculator)
        self.drift: Dict[str, float] = {d: 0.0 for d in domains}

        # A4: Coverage compensation state
        self.gap: Dict[str, int] = {d: 0 for d in domains}

        # Domain sample counts
        self.sample_count: Dict[str, int] = {d: 0 for d in domains}

        # Previous round aggregation weights (for gap computation)
        self._prev_weights: Dict[str, float] = {d: 1.0 / len(domains) for d in domains}

    def update_loss(self, domain: str, val_loss: float) -> float:
        """Update EMA validation loss for a domain (A2).

        Args:
            domain: Domain name
            val_loss: Raw validation loss L_raw_e(t)

        Returns:
            Updated EMA loss L_bar_e(t)
        """
        return self.L_ema[domain].update(val_loss)

    def set_drift(self, domain: str, drift_score: float) -> None:
        """Set drift score for a domain (computed externally by DriftCalculator).

        Args:
            domain: Domain name
            drift_score: Drift score Delta_e(t)
        """
        self.drift[domain] = drift_score

    def set_sample_count(self, domain: str, count: int) -> None:
        """Set sample count for a domain.

        Args:
            domain: Domain name
            count: Total samples n_e = sum_{k in K_e} n_k
        """
        self.sample_count[domain] = count

    def update_coverage_gap(self, current_weights: Dict[str, float]) -> None:
        """Update coverage compensation counters based on previous round weights (A4).

        Formula:
            gap_e(t) = 0                    if alpha_e(t-1) >= alpha_min
                     = gap_e(t-1) + 1       if alpha_e(t-1) < alpha_min

        Args:
            current_weights: Aggregation weights from previous round {domain: alpha_e}
        """
        for domain in self.domains:
            prev_weight = self._prev_weights.get(domain, 1.0 / len(self.domains))
            if prev_weight >= self.alpha_min:
                self.gap[domain] = 0
            else:
                self.gap[domain] += 1

        # Store for next round
        self._prev_weights = current_weights.copy()

    def get_coverage_compensation(self, domain: str) -> float:
        """Compute coverage compensation factor H_e(t) for a domain (A4).

        Formula:
            H_e(t) = tanh(gap_e(t) / T_H)

        Args:
            domain: Domain name

        Returns:
            Coverage compensation H_e in [0, 1)
        """
        return math.tanh(self.gap[domain] / self.T_H)

    def get_all_states(self) -> Dict[str, Dict[str, float]]:
        """Get all domain states for cloud aggregation.

        Returns:
            Dictionary with keys:
                - L_map: {domain: EMA_loss}
                - drift_map: {domain: drift_score}
                - H_map: {domain: coverage_compensation}
                - n_map: {domain: sample_count}
        """
        L_map = {}
        drift_map = {}
        H_map = {}
        n_map = {}

        for domain in self.domains:
            # EMA loss (default to 0 if not yet computed)
            L_val = self.L_ema[domain].get()
            L_map[domain] = L_val if L_val is not None else 0.0

            # Drift score
            drift_map[domain] = self.drift[domain]

            # Coverage compensation
            H_map[domain] = self.get_coverage_compensation(domain)

            # Sample count
            n_map[domain] = self.sample_count[domain]

        return {
            'L_map': L_map,
            'drift_map': drift_map,
            'H_map': H_map,
            'n_map': n_map
        }

    def reset(self) -> None:
        """Reset all state trackers."""
        for domain in self.domains:
            self.L_ema[domain].reset()
            self.drift[domain] = 0.0
            self.gap[domain] = 0
            self.sample_count[domain] = 0
        self._prev_weights = {d: 1.0 / len(self.domains) for d in self.domains}
