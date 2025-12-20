"""FAP v3 Fair Aggregation Module.

This module provides enhanced fair-weighted aggregation for federated learning
with domain shift, implementing a three-tier architecture:

DC/Edge Side (domain_state.py, drift_calculator.py):
    - A1: Per-domain state management
    - A2: EMA validation loss tracking
    - A3: S1 statistic aggregation with random projection for drift computation
    - A4: Coverage compensation for long-term fairness

Cloud Side (cloud_aggregator.py):
    - B2: Z-score normalization of domain states
    - B3: Score computation (loss compensation + drift penalty + coverage)
    - B4: Sample-weight fusion with clipping and EMA smoothing
    - B5: Weighted aggregation of global backbone theta

Key Design Principles:
    - Higher loss → Higher weight (domains needing more attention)
    - Higher drift → Lower weight (unstable contributions may be harmful)
    - Coverage compensation → Prevent long-term starvation of any domain

Example Usage:
    >>> from core.fair_aggregation import (
    ...     DomainStateManager,
    ...     DriftCalculator,
    ...     CloudAggregator
    ... )
    >>>
    >>> # Initialize components
    >>> domains = ['photo', 'art', 'cartoon', 'sketch']
    >>> state_mgr = DomainStateManager(domains)
    >>> drift_calc = DriftCalculator(feature_dim=512)
    >>> cloud_agg = CloudAggregator(domains)
    >>>
    >>> # During training loop (DC side)
    >>> state_mgr.update_loss('photo', val_loss)
    >>> state_mgr.set_drift('photo', drift_calc.compute_drift('photo'))
    >>>
    >>> # At aggregation time (Cloud side)
    >>> states = state_mgr.get_all_states()
    >>> weights = cloud_agg.compute_aggregation_weights(**states)

Configuration:
    The module behavior can be controlled via config:
    ```yaml
    fair_aggregation:
        enable: true  # Master switch (default: enabled)
        w1: 1.0        # Loss compensation weight
        w2: 0.3        # Drift penalty weight
        w3: 0.5        # Coverage compensation weight
        tau: 1.0       # Softmax temperature
        gamma: 0.5     # Sample weight exponent
        alpha_floor: 0.05  # Minimum weight per domain
        alpha_ceil: 0.7    # Maximum weight per domain
        ema_beta: 0.8      # Weight EMA smoothing
    ```
"""

from .domain_state import DomainStateManager, EMATracker
from .drift_calculator import DriftCalculator, ClientStatistics
from .cloud_aggregator import CloudAggregator

__all__ = [
    # Domain state management (A1, A2, A4)
    'DomainStateManager',
    'EMATracker',

    # Drift calculation (A3)
    'DriftCalculator',
    'ClientStatistics',

    # Cloud aggregation (B2-B5)
    'CloudAggregator',
]

__version__ = '3.0.0'
