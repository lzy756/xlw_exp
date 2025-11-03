"""Unit tests for aggregation functions."""

import pytest
import torch
from core.aggregator import fedavg, aggregate_theta, aggregate_phi_domain


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