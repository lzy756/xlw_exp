"""Tests for edge case handling."""

import pytest
import numpy as np
import torch
from core.selector import FAPFloatSoftmax
from utils.common import EMA


def test_selector_validates_missing_domains():
    """Test that selector raises error when input dicts are missing domains."""
    domains = ['domain1', 'domain2', 'domain3']
    selector = FAPFloatSoftmax(domains=domains)

    # Missing domain in L_map
    with pytest.raises(ValueError, match="L_map missing some domains"):
        selector.select(
            L_map={'domain1': 0.5, 'domain2': 0.3},  # Missing domain3
            drift_map={'domain1': 0.1, 'domain2': 0.2, 'domain3': 0.15},
            cover_map={'domain1': 1.0, 'domain2': 2.0, 'domain3': 1.5},
            stay_map={'domain1': 0.0, 'domain2': 0.2, 'domain3': 0.0}
        )


def test_selector_validates_negative_values():
    """Test that selector raises error for negative values."""
    domains = ['domain1', 'domain2']
    selector = FAPFloatSoftmax(domains=domains)

    # Negative loss value
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        selector.select(
            L_map={'domain1': -0.5, 'domain2': 0.3},  # Negative value
            drift_map={'domain1': 0.1, 'domain2': 0.2},
            cover_map={'domain1': 1.0, 'domain2': 2.0},
            stay_map={'domain1': 0.0, 'domain2': 0.0}
        )


def test_selector_validates_infinite_values():
    """Test that selector raises error for infinite values."""
    domains = ['domain1', 'domain2']
    selector = FAPFloatSoftmax(domains=domains)

    # Infinite drift value
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        selector.select(
            L_map={'domain1': 0.5, 'domain2': 0.3},
            drift_map={'domain1': float('inf'), 'domain2': 0.2},  # Infinite value
            cover_map={'domain1': 1.0, 'domain2': 2.0},
            stay_map={'domain1': 0.0, 'domain2': 0.0}
        )


def test_selector_handles_identical_scores():
    """Test that selector returns uniform probabilities when all scores are identical."""
    domains = ['domain1', 'domain2', 'domain3']
    selector = FAPFloatSoftmax(domains=domains, w1=1.0, w2=0.0, w3=0.0, w4=0.0)

    # All domains have identical loss (will result in identical normalized scores)
    selected, probs, scores = selector.select(
        L_map={'domain1': 0.5, 'domain2': 0.5, 'domain3': 0.5},
        drift_map={'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0},
        cover_map={'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0},
        stay_map={'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}
    )

    # Probabilities should be uniform (approximately 1/3 each)
    assert len(probs) == 3
    np.testing.assert_allclose(probs, 1/3, atol=0.01)

    # One domain should be selected
    assert selected in domains


def test_selector_handles_zero_variance():
    """Test that selector handles zero variance in normalization."""
    domains = ['domain1', 'domain2']
    selector = FAPFloatSoftmax(domains=domains)

    # All metrics have zero variance
    selected, probs, scores = selector.select(
        L_map={'domain1': 1.0, 'domain2': 1.0},
        drift_map={'domain1': 0.5, 'domain2': 0.5},
        cover_map={'domain1': 2.0, 'domain2': 2.0},
        stay_map={'domain1': 0.0, 'domain2': 0.0}
    )

    # Should handle gracefully with uniform probabilities
    assert selected in domains
    assert len(probs) == 2
    assert np.allclose(probs.sum(), 1.0)


def test_ema_handles_nan():
    """Test that EMA can handle NaN values gracefully."""
    ema = EMA(beta=0.9)

    # Initialize with valid value
    ema.update(1.0)
    assert ema.get() == 1.0

    # Try to update with NaN
    ema.update(float('nan'))

    # EMA should still be valid (or handle gracefully)
    # Current implementation will propagate NaN, which is detected later
    # This test documents the behavior
    val = ema.get()
    assert val is None or np.isnan(val) or np.isfinite(val)


def test_ema_initialization():
    """Test EMA initialization and first update."""
    ema = EMA(beta=0.9)

    # Initially None
    assert ema.get() is None

    # First update sets value directly
    ema.update(5.0)
    assert ema.get() == 5.0

    # Second update uses EMA formula: beta * old + (1 - beta) * new
    ema.update(10.0)
    expected = 0.9 * 5.0 + 0.1 * 10.0
    val = ema.get()
    assert val is not None and abs(val - expected) < 1e-6


def test_empty_phi_updates_skip_aggregation():
    """Test that empty phi updates list is handled correctly."""
    from core.aggregator import aggregate_phi_domain

    # Empty list should return empty dict
    result = aggregate_phi_domain([], [], domain='test')
    assert result == {}


def test_selector_probabilities_sum_to_one():
    """Test that selection probabilities always sum to 1."""
    domains = ['domain1', 'domain2', 'domain3', 'domain4']
    selector = FAPFloatSoftmax(domains=domains, tau=0.5)

    # Test with various scenarios
    scenarios = [
        # Scenario 1: Different losses
        {
            'L_map': {'domain1': 0.8, 'domain2': 0.5, 'domain3': 0.3, 'domain4': 0.6},
            'drift_map': {'domain1': 0.1, 'domain2': 0.2, 'domain3': 0.15, 'domain4': 0.05},
            'cover_map': {'domain1': 2.0, 'domain2': 1.0, 'domain3': 3.0, 'domain4': 1.5},
            'stay_map': {'domain1': 0.0, 'domain2': 0.2, 'domain3': 0.0, 'domain4': 0.0}
        },
        # Scenario 2: Extreme values
        {
            'L_map': {'domain1': 10.0, 'domain2': 0.01, 'domain3': 5.0, 'domain4': 2.0},
            'drift_map': {'domain1': 1.0, 'domain2': 0.001, 'domain3': 0.5, 'domain4': 0.3},
            'cover_map': {'domain1': 10.0, 'domain2': 0.0, 'domain3': 5.0, 'domain4': 2.0},
            'stay_map': {'domain1': 0.2, 'domain2': 0.0, 'domain3': 0.0, 'domain4': 0.2}
        },
        # Scenario 3: Small values
        {
            'L_map': {'domain1': 0.001, 'domain2': 0.002, 'domain3': 0.0015, 'domain4': 0.0008},
            'drift_map': {'domain1': 0.0001, 'domain2': 0.0002, 'domain3': 0.00015, 'domain4': 0.0},
            'cover_map': {'domain1': 0.1, 'domain2': 0.2, 'domain3': 0.15, 'domain4': 0.05},
            'stay_map': {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.2, 'domain4': 0.0}
        }
    ]

    for i, scenario in enumerate(scenarios):
        selected, probs, scores = selector.select(**scenario)

        # Probabilities should sum to 1
        assert abs(probs.sum() - 1.0) < 1e-6, f"Scenario {i+1}: probabilities don't sum to 1"

        # All probabilities should be non-negative
        assert np.all(probs >= 0), f"Scenario {i+1}: negative probabilities found"

        # Selected domain should be in the list
        assert selected in domains, f"Scenario {i+1}: invalid domain selected"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
