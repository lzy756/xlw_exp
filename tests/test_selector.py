"""Unit tests for FAPFloatSoftmax selector."""

import pytest
import numpy as np
from core.selector import FAPFloatSoftmax


def test_uniform_weights_uniform_probs():
    """Test that uniform inputs with uniform weights give uniform probabilities."""
    domains = ['domain1', 'domain2', 'domain3']
    selector = FAPFloatSoftmax(
        domains=domains,
        w1=1.0,
        w2=1.0,
        w3=1.0,
        w4=0.0,  # No stability bonus
        tau=1.0
    )

    # All metrics identical
    L_map = {'domain1': 0.5, 'domain2': 0.5, 'domain3': 0.5}
    drift_map = {'domain1': 0.1, 'domain2': 0.1, 'domain3': 0.1}
    cover_map = {'domain1': 1.0, 'domain2': 1.0, 'domain3': 1.0}
    stay_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}

    selected, probs, scores = selector.select(L_map, drift_map, cover_map, stay_map)

    # Probabilities should be uniform (approximately)
    expected_prob = 1.0 / 3.0
    for prob in probs:
        assert abs(prob - expected_prob) < 0.01

    # Selected domain should be one of the domains
    assert selected in domains

    # Scores should be all zeros (after normalization)
    for score in scores:
        assert abs(score) < 1e-6


def test_high_loss_increases_probability():
    """Test that domains with higher loss get higher selection probability."""
    domains = ['domain1', 'domain2', 'domain3']
    selector = FAPFloatSoftmax(
        domains=domains,
        w1=1.0,  # Weight loss highly
        w2=0.0,
        w3=0.0,
        w4=0.0,
        tau=1.0
    )

    # domain3 has much higher loss
    L_map = {'domain1': 0.1, 'domain2': 0.1, 'domain3': 1.0}
    drift_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}
    cover_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}
    stay_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}

    selected, probs, scores = selector.select(L_map, drift_map, cover_map, stay_map)

    # domain3 should have highest probability
    domain3_idx = domains.index('domain3')
    domain1_idx = domains.index('domain1')
    domain2_idx = domains.index('domain2')

    assert probs[domain3_idx] > probs[domain1_idx]
    assert probs[domain3_idx] > probs[domain2_idx]

    # domain3 should have highest score
    assert scores[domain3_idx] > scores[domain1_idx]
    assert scores[domain3_idx] > scores[domain2_idx]


def test_probabilities_sum_to_one():
    """Test that selection probabilities always sum to 1."""
    domains = ['domain1', 'domain2', 'domain3', 'domain4']
    selector = FAPFloatSoftmax(
        domains=domains,
        w1=1.0,
        w2=0.5,
        w3=0.3,
        w4=0.2,
        tau=1.0
    )

    # Various random metrics
    L_map = {'domain1': 0.5, 'domain2': 0.8, 'domain3': 0.3, 'domain4': 0.6}
    drift_map = {'domain1': 0.2, 'domain2': 0.1, 'domain3': 0.4, 'domain4': 0.15}
    cover_map = {'domain1': 2.0, 'domain2': 1.0, 'domain3': 0.0, 'domain4': 3.0}
    stay_map = {'domain1': 0.0, 'domain2': 0.2, 'domain3': 0.0, 'domain4': 0.0}

    selected, probs, scores = selector.select(L_map, drift_map, cover_map, stay_map)

    # Probabilities should sum to 1
    assert abs(probs.sum() - 1.0) < 1e-6

    # All probabilities should be non-negative
    assert np.all(probs >= 0)


def test_stability_bonus_continuity():
    """Test that stability bonus increases probability for staying with same aggregator."""
    domains = ['domain1', 'domain2', 'domain3']
    selector = FAPFloatSoftmax(
        domains=domains,
        w1=0.0,
        w2=0.0,
        w3=0.0,
        w4=1.0,  # Only stability bonus matters
        tau=1.0
    )

    # All metrics same except stability
    L_map = {'domain1': 0.5, 'domain2': 0.5, 'domain3': 0.5}
    drift_map = {'domain1': 0.1, 'domain2': 0.1, 'domain3': 0.1}
    cover_map = {'domain1': 1.0, 'domain2': 1.0, 'domain3': 1.0}

    # Case 1: No stability bonus
    stay_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}
    selected1, probs1, scores1 = selector.select(L_map, drift_map, cover_map, stay_map)

    # All probabilities should be similar
    assert abs(probs1[0] - probs1[1]) < 0.01
    assert abs(probs1[1] - probs1[2]) < 0.01

    # Case 2: domain2 has stability bonus
    stay_map = {'domain1': 0.0, 'domain2': 0.2, 'domain3': 0.0}
    selected2, probs2, scores2 = selector.select(L_map, drift_map, cover_map, stay_map)

    # domain2 should have higher probability
    domain2_idx = domains.index('domain2')
    domain1_idx = domains.index('domain1')
    domain3_idx = domains.index('domain3')

    assert probs2[domain2_idx] > probs2[domain1_idx]
    assert probs2[domain2_idx] > probs2[domain3_idx]


def test_low_tau_more_deterministic():
    """Test that lower temperature makes selection more deterministic."""
    domains = ['domain1', 'domain2', 'domain3']

    # High temperature (more random)
    selector_high_tau = FAPFloatSoftmax(
        domains=domains,
        w1=1.0,
        w2=0.0,
        w3=0.0,
        w4=0.0,
        tau=10.0  # High temperature
    )

    # Low temperature (more deterministic)
    selector_low_tau = FAPFloatSoftmax(
        domains=domains,
        w1=1.0,
        w2=0.0,
        w3=0.0,
        w4=0.0,
        tau=0.1  # Low temperature
    )

    # domain3 has much higher loss
    L_map = {'domain1': 0.1, 'domain2': 0.1, 'domain3': 1.0}
    drift_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}
    cover_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}
    stay_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}

    # High tau selection
    _, probs_high_tau, _ = selector_high_tau.select(L_map, drift_map, cover_map, stay_map)

    # Low tau selection
    _, probs_low_tau, _ = selector_low_tau.select(L_map, drift_map, cover_map, stay_map)

    # With low tau, domain3 should have much higher probability (more deterministic)
    # With high tau, probabilities should be more spread out (less deterministic)
    domain3_idx = domains.index('domain3')

    # Low tau should give domain3 a higher probability than high tau
    assert probs_low_tau[domain3_idx] > probs_high_tau[domain3_idx]

    # Low tau should be closer to 1 (more deterministic)
    assert probs_low_tau[domain3_idx] > 0.9  # Should be very high


def test_selector_input_validation():
    """Test that selector validates input dictionaries."""
    domains = ['domain1', 'domain2']
    selector = FAPFloatSoftmax(
        domains=domains,
        w1=1.0,
        w2=0.5,
        w3=0.3,
        w4=0.2,
        tau=1.0
    )

    # Missing domain in L_map
    with pytest.raises(ValueError, match="L_map missing some domains"):
        selector.select(
            L_map={'domain1': 0.5},  # Missing domain2
            drift_map={'domain1': 0.1, 'domain2': 0.1},
            cover_map={'domain1': 1.0, 'domain2': 1.0},
            stay_map={'domain1': 0.0, 'domain2': 0.0}
        )

    # Negative value
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        selector.select(
            L_map={'domain1': -0.5, 'domain2': 0.5},  # Negative value
            drift_map={'domain1': 0.1, 'domain2': 0.1},
            cover_map={'domain1': 1.0, 'domain2': 1.0},
            stay_map={'domain1': 0.0, 'domain2': 0.0}
        )

    # NaN value
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        selector.select(
            L_map={'domain1': float('nan'), 'domain2': 0.5},  # NaN value
            drift_map={'domain1': 0.1, 'domain2': 0.1},
            cover_map={'domain1': 1.0, 'domain2': 1.0},
            stay_map={'domain1': 0.0, 'domain2': 0.0}
        )


def test_selector_handles_zero_variance():
    """Test that selector handles case when all scores are identical."""
    domains = ['domain1', 'domain2', 'domain3']
    selector = FAPFloatSoftmax(
        domains=domains,
        w1=1.0,
        w2=1.0,
        w3=1.0,
        w4=0.0,
        tau=1.0
    )

    # All metrics identical (zero variance after normalization)
    L_map = {'domain1': 0.5, 'domain2': 0.5, 'domain3': 0.5}
    drift_map = {'domain1': 0.1, 'domain2': 0.1, 'domain3': 0.1}
    cover_map = {'domain1': 1.0, 'domain2': 1.0, 'domain3': 1.0}
    stay_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}

    # Should not raise error
    selected, probs, scores = selector.select(L_map, drift_map, cover_map, stay_map)

    # Should return uniform probabilities
    expected_prob = 1.0 / 3.0
    for prob in probs:
        assert abs(prob - expected_prob) < 0.01

    # Selected should be valid
    assert selected in domains


def test_weighted_combination():
    """Test that selector properly combines multiple metrics with weights."""
    domains = ['domain1', 'domain2', 'domain3']
    selector = FAPFloatSoftmax(
        domains=domains,
        w1=1.0,
        w2=1.0,
        w3=1.0,
        w4=0.0,
        tau=1.0
    )

    # domain1: high loss, low drift, low coverage
    # domain2: low loss, high drift, low coverage
    # domain3: low loss, low drift, high coverage
    L_map = {'domain1': 1.0, 'domain2': 0.1, 'domain3': 0.1}
    drift_map = {'domain1': 0.1, 'domain2': 1.0, 'domain3': 0.1}
    cover_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 5.0}
    stay_map = {'domain1': 0.0, 'domain2': 0.0, 'domain3': 0.0}

    selected, probs, scores = selector.select(L_map, drift_map, cover_map, stay_map)

    # All domains should have non-zero probability
    assert np.all(probs > 0)

    # Probabilities should sum to 1
    assert abs(probs.sum() - 1.0) < 1e-6
