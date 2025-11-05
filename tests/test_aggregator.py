"""Unit tests for aggregation functions."""

import pytest
import torch
import numpy as np
from core.aggregator import (
    fedavg, 
    aggregate_theta, 
    aggregate_phi_domain,
    _normalize_scores,
    aggregate_theta_weighted
)


def test_fedavg_weighted_average():
    """Test that FedAvg computes correct weighted average."""
    # Create dummy state dicts
    state1 = {'param': torch.tensor([1.0, 2.0, 3.0])}
    state2 = {'param': torch.tensor([4.0, 5.0, 6.0])}
    state3 = {'param': torch.tensor([7.0, 8.0, 9.0])}

    states = [state1, state2, state3]
    weights = [0.2, 0.3, 0.5]  # Weights sum to 1.0

    # Aggregate
    result = fedavg(states, weights)

    # Expected: 0.2*[1,2,3] + 0.3*[4,5,6] + 0.5*[7,8,9]
    expected = torch.tensor([4.9, 5.9, 6.9])

    assert torch.allclose(result['param'], expected, atol=1e-6)


def test_aggregate_theta_correctness():
    """Test that theta aggregation excludes phi parameters."""
    # Create state dicts with mixed parameters
    state1 = {
        'conv1.weight': torch.ones(3, 3),
        'heads.clipart.weight': torch.ones(10, 512),  # Should be excluded
        'lora_blocks.layer4_0.weight': torch.ones(16, 512),  # Should be excluded
        'bn1.weight': torch.ones(64)
    }

    state2 = {
        'conv1.weight': torch.ones(3, 3) * 2,
        'heads.clipart.weight': torch.ones(10, 512) * 2,
        'lora_blocks.layer4_0.weight': torch.ones(16, 512) * 2,
        'bn1.weight': torch.ones(64) * 2
    }

    states = [state1, state2]
    weights = [0.5, 0.5]

    # Aggregate theta
    result = aggregate_theta(states, weights)

    # Check that only theta parameters are included
    assert 'conv1.weight' in result
    assert 'bn1.weight' in result
    assert 'heads.clipart.weight' not in result
    assert 'lora_blocks.layer4_0.weight' not in result

    # Check values
    assert torch.allclose(result['conv1.weight'], torch.ones(3, 3) * 1.5)
    assert torch.allclose(result['bn1.weight'], torch.ones(64) * 1.5)


def test_aggregate_phi_domain_filtering():
    """Test that phi aggregation includes only domain-specific parameters."""
    # Create state dicts with phi parameters
    state1 = {
        'conv1.weight': torch.ones(3, 3),  # Should be excluded
        'heads.clipart.weight': torch.ones(10, 512),
        'heads.clipart.bias': torch.ones(10),
        'lora_blocks.layer4_0.weight': torch.ones(16, 512),
        'bn1.weight': torch.ones(64)  # Should be excluded
    }

    state2 = {
        'conv1.weight': torch.ones(3, 3) * 2,
        'heads.clipart.weight': torch.ones(10, 512) * 2,
        'heads.clipart.bias': torch.ones(10) * 2,
        'lora_blocks.layer4_0.weight': torch.ones(16, 512) * 2,
        'bn1.weight': torch.ones(64) * 2
    }

    states = [state1, state2]
    weights = [0.3, 0.7]

    # Aggregate phi for clipart domain
    result = aggregate_phi_domain(states, weights, domain='clipart')

    # Check that only phi parameters are included
    assert 'heads.clipart.weight' in result
    assert 'heads.clipart.bias' in result
    assert 'lora_blocks.layer4_0.weight' in result
    assert 'conv1.weight' not in result
    assert 'bn1.weight' not in result

    # Check values (0.3 * 1 + 0.7 * 2 = 1.7)
    assert torch.allclose(result['heads.clipart.weight'], torch.ones(10, 512) * 1.7)
    assert torch.allclose(result['lora_blocks.layer4_0.weight'], torch.ones(16, 512) * 1.7)


def test_fedavg_empty_input():
    """Test FedAvg with empty input."""
    result = fedavg([], [])
    assert result == {}


def test_fedavg_weight_normalization():
    """Test that FedAvg normalizes weights correctly."""
    state1 = {'param': torch.tensor([1.0])}
    state2 = {'param': torch.tensor([2.0])}

    states = [state1, state2]
    weights = [2.0, 3.0]  # Sum is 5.0, not 1.0

    result = fedavg(states, weights)

    # Should normalize: 2/5 * 1 + 3/5 * 2 = 0.4 + 1.2 = 1.6
    expected = torch.tensor([1.6])
    assert torch.allclose(result['param'], expected, atol=1e-6)


def test_aggregate_phi_excludes_theta_keys():
    """Test that aggregate_phi_domain excludes backbone (theta) parameters."""
    # Create state dicts with all types of parameters
    state1 = {
        'conv1.weight': torch.ones(64, 3, 7, 7),  # Theta - should be excluded
        'layer1.0.conv1.weight': torch.ones(64, 64, 3, 3),  # Theta - should be excluded
        'fc.weight': torch.ones(512, 1000),  # Theta - should be excluded
        'lora_blocks.0.weight': torch.ones(16, 512),  # Phi - should be included
        'heads.domain1.weight': torch.ones(10, 512),  # Phi - should be included
    }

    state2 = {k: v * 2 for k, v in state1.items()}

    states = [state1, state2]
    weights = [0.5, 0.5]

    # Aggregate phi for domain1
    result = aggregate_phi_domain(states, weights, domain='domain1')

    # Verify only phi parameters are included
    assert 'lora_blocks.0.weight' in result, "LoRA blocks should be in phi"
    assert 'heads.domain1.weight' in result, "Domain head should be in phi"
    assert 'conv1.weight' not in result, "Backbone conv should not be in phi"
    assert 'layer1.0.conv1.weight' not in result, "Backbone layer should not be in phi"
    assert 'fc.weight' not in result, "Backbone FC should not be in phi"


def test_aggregate_theta_excludes_phi_keys():
    """Test that aggregate_theta excludes LoRA and head (phi) parameters."""
    # Create state dicts with all types of parameters
    state1 = {
        'conv1.weight': torch.ones(64, 3, 7, 7),  # Theta - should be included
        'layer1.0.conv1.weight': torch.ones(64, 64, 3, 3),  # Theta - should be included
        'bn1.weight': torch.ones(64),  # Theta - should be included
        'lora_blocks.0.weight': torch.ones(16, 512),  # Phi - should be excluded
        'lora_blocks.0.bias': torch.ones(16),  # Phi - should be excluded
        'heads.domain1.weight': torch.ones(10, 512),  # Phi - should be excluded
        'heads.domain2.weight': torch.ones(10, 512),  # Phi - should be excluded
    }

    state2 = {k: v * 2 for k, v in state1.items()}

    states = [state1, state2]
    weights = [0.5, 0.5]

    # Aggregate theta
    result = aggregate_theta(states, weights)

    # Verify only theta parameters are included
    assert 'conv1.weight' in result, "Backbone conv should be in theta"
    assert 'layer1.0.conv1.weight' in result, "Backbone layer should be in theta"
    assert 'bn1.weight' in result, "Backbone BN should be in theta"
    assert 'lora_blocks.0.weight' not in result, "LoRA blocks should not be in theta"
    assert 'lora_blocks.0.bias' not in result, "LoRA bias should not be in theta"
    assert 'heads.domain1.weight' not in result, "Domain1 head should not be in theta"
    assert 'heads.domain2.weight' not in result, "Domain2 head should not be in theta"


# ============================================================================
# Tests for Fair-Weighted Aggregation
# ============================================================================

def test_normalize_scores_zscore():
    """Test z-score normalization of scores."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    normalized = _normalize_scores(scores, mode='zscore')
    
    # Z-score should have mean=0, std=1
    assert abs(np.mean(normalized)) < 1e-6, "Z-score mean should be ~0"
    assert abs(np.std(normalized) - 1.0) < 1e-6, "Z-score std should be ~1"


def test_normalize_scores_minmax():
    """Test min-max normalization of scores."""
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    normalized = _normalize_scores(scores, mode='minmax')
    
    # Min-max should be in [0, 1]
    assert normalized.min() == 0.0, "Min should be 0"
    assert normalized.max() == 1.0, "Max should be 1"
    assert np.array_equal(normalized, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_normalize_scores_zero_variance():
    """Test normalization with zero variance (all same values)."""
    scores = np.array([2.0, 2.0, 2.0])
    
    # Z-score with zero variance should return zeros
    normalized_z = _normalize_scores(scores, mode='zscore')
    assert np.allclose(normalized_z, np.zeros(3))
    
    # Min-max with zero range should return zeros
    normalized_mm = _normalize_scores(scores, mode='minmax')
    assert np.allclose(normalized_mm, np.zeros(3))


def test_normalize_scores_invalid_mode():
    """Test that invalid normalization mode raises error."""
    scores = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        _normalize_scores(scores, mode='invalid')


def test_aggregate_theta_weighted_basic():
    """Test basic fair-weighted aggregation."""
    # Create dummy theta states for 3 domains
    theta1 = {'conv.weight': torch.tensor([1.0, 1.0, 1.0])}
    theta2 = {'conv.weight': torch.tensor([2.0, 2.0, 2.0])}
    theta3 = {'conv.weight': torch.tensor([3.0, 3.0, 3.0])}
    
    theta_list = [theta1, theta2, theta3]
    domains = ['d1', 'd2', 'd3']
    n_map = {'d1': 100, 'd2': 200, 'd3': 300}  # Sample counts
    score_map = {'d1': 0.5, 'd2': 0.6, 'd3': 0.7}  # Higher score = worse performance
    agg_domain = 'd2'
    
    config = {
        'fair_weighting': {
            'beta': 0.5,
            'lambda_boost': 0.2,
            'norm': 'zscore'
        }
    }
    
    theta_avg, alpha, q = aggregate_theta_weighted(
        theta_list, domains, n_map, score_map, agg_domain, config
    )
    
    # Check outputs
    assert 'conv.weight' in theta_avg
    assert len(alpha) == 3
    assert len(q) == 3
    assert np.allclose(np.sum(alpha), 1.0), "Alpha should sum to 1"
    assert np.all(alpha >= 0), "Alpha should be non-negative"
    assert np.all(q >= 0), "Fairness factors should be non-negative"
    assert np.allclose(np.sum(q), 1.0), "Fairness factors should sum to 1"


def test_aggregate_theta_weighted_lambda_boost():
    """Test that lambda boost increases selected aggregator's weight."""
    # Create identical theta states
    theta = {'param': torch.tensor([1.0])}
    theta_list = [theta.copy(), theta.copy(), theta.copy()]
    
    domains = ['d1', 'd2', 'd3']
    n_map = {'d1': 100, 'd2': 100, 'd3': 100}  # Equal samples
    score_map = {'d1': 1.0, 'd2': 1.0, 'd3': 1.0}  # Equal scores
    
    # Test with d2 selected
    config = {
        'fair_weighting': {
            'beta': 0.5,
            'lambda_boost': 0.5,
            'norm': 'zscore'
        }
    }
    
    _, alpha, q = aggregate_theta_weighted(
        theta_list, domains, n_map, score_map, 'd2', config
    )
    
    # Lambda boost adds to d2's normalized score -> higher fairness factor -> higher weight
    d2_idx = domains.index('d2')
    assert alpha[d2_idx] > alpha[0] and alpha[d2_idx] > alpha[2], \
        "Selected aggregator should have higher weight due to lambda boost"


def test_aggregate_theta_weighted_fairness_effect():
    """Test that fairness factors favor struggling domains (high scores)."""
    theta = {'param': torch.tensor([1.0])}
    theta_list = [theta.copy(), theta.copy(), theta.copy()]
    
    domains = ['d1', 'd2', 'd3']
    n_map = {'d1': 100, 'd2': 100, 'd3': 100}  # Equal samples
    # d3 has highest score (worst performance), should get higher fairness factor
    score_map = {'d1': 0.3, 'd2': 0.5, 'd3': 0.9}
    
    config = {
        'fair_weighting': {
            'beta': 1.0,  # Higher beta = more emphasis on fairness
            'lambda_boost': 0.0,  # No boost to isolate fairness effect
            'norm': 'zscore'
        }
    }
    
    _, alpha, q = aggregate_theta_weighted(
        theta_list, domains, n_map, score_map, 'd1', config
    )
    
    # Domain with higher score (d3=0.9) should get higher fairness factor
    # After zscore normalization and softmax with positive beta
    assert q[2] > q[0] and q[2] > q[1], \
        "Worst-performing domain (highest score) should have highest fairness factor"


def test_aggregate_theta_weighted_beta_effect():
    """Test that beta controls temperature of fairness softmax."""
    theta = {'param': torch.tensor([1.0])}
    theta_list = [theta.copy(), theta.copy()]
    
    domains = ['d1', 'd2']
    n_map = {'d1': 100, 'd2': 100}
    score_map = {'d1': 0.2, 'd2': 0.8}  # Large difference
    
    # Low beta (more uniform)
    config_low = {
        'fair_weighting': {
            'beta': 0.1,
            'lambda_boost': 0.0,
            'norm': 'zscore'
        }
    }
    
    _, _, q_low = aggregate_theta_weighted(
        theta_list, domains, n_map, score_map, 'd1', config_low
    )
    
    # High beta (sharper differences)
    config_high = {
        'fair_weighting': {
            'beta': 2.0,
            'lambda_boost': 0.0,
            'norm': 'zscore'
        }
    }
    
    _, _, q_high = aggregate_theta_weighted(
        theta_list, domains, n_map, score_map, 'd1', config_high
    )
    
    # Higher beta should create larger difference between q values
    diff_low = abs(q_low[1] - q_low[0])
    diff_high = abs(q_high[1] - q_high[0])
    assert diff_high > diff_low, "Higher beta should create sharper differences"


def test_aggregate_theta_weighted_input_validation():
    """Test input validation for fair-weighted aggregation."""
    theta = {'param': torch.tensor([1.0])}
    theta_list = [theta.copy()]
    domains = ['d1']
    
    config = {
        'fair_weighting': {
            'beta': 0.5,
            'lambda_boost': 0.2,
            'norm': 'zscore'
        }
    }
    
    # Missing domain in n_map
    with pytest.raises(ValueError, match="missing from n_map"):
        aggregate_theta_weighted(
            theta_list, domains, {}, {'d1': 1.0}, 'd1', config
        )
    
    # Missing domain in score_map
    with pytest.raises(ValueError, match="missing from score_map"):
        aggregate_theta_weighted(
            theta_list, domains, {'d1': 100}, {}, 'd1', config
        )
    
    # Negative sample count
    with pytest.raises(ValueError, match="must be non-negative"):
        aggregate_theta_weighted(
            theta_list, domains, {'d1': -10}, {'d1': 1.0}, 'd1', config
        )
    
    # Non-finite scores
    with pytest.raises(ValueError, match="must be finite"):
        aggregate_theta_weighted(
            theta_list, domains, {'d1': 100}, {'d1': float('inf')}, 'd1', config
        )


def test_aggregate_theta_weighted_weight_combination():
    """Test that final weights combine sample counts and fairness factors correctly."""
    theta = {'param': torch.tensor([1.0])}
    theta_list = [theta.copy(), theta.copy()]
    
    domains = ['d1', 'd2']
    # d1 has more samples but worse performance (higher score)
    n_map = {'d1': 300, 'd2': 100}
    score_map = {'d1': 0.8, 'd2': 0.2}  # d1 worse
    
    config = {
        'fair_weighting': {
            'beta': 1.0,
            'lambda_boost': 0.0,
            'norm': 'zscore'
        }
    }
    
    _, alpha, q = aggregate_theta_weighted(
        theta_list, domains, n_map, score_map, 'd1', config
    )
    
    # Verify alpha is normalized product of n and q
    n = np.array([n_map[d] for d in domains])
    expected_unnorm = n * q
    expected_alpha = expected_unnorm / np.sum(expected_unnorm)
    
    assert np.allclose(alpha, expected_alpha, atol=1e-6), \
        "Alpha should be normalized product of sample counts and fairness factors"