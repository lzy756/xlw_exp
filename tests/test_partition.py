"""Unit tests for data partitioning."""

import pytest
import numpy as np
from data.partition import dirichlet_split_by_class, build_domain_clients


def test_dirichlet_split_totals_match():
    """Test that Dirichlet split preserves all samples."""
    # Create dummy labels
    n_samples = 1000
    n_classes = 10
    n_clients = 5
    labels = np.random.randint(0, n_classes, n_samples)

    # Perform split
    client_indices = dirichlet_split_by_class(
        labels=labels,
        num_clients=n_clients,
        alpha=0.1,
        seed=42
    )

    # Check total samples preserved
    total_assigned = sum(len(indices) for indices in client_indices)
    assert total_assigned == n_samples, f"Lost samples: {n_samples} -> {total_assigned}"

    # Check no duplicate assignments
    all_indices = []
    for indices in client_indices:
        all_indices.extend(indices)
    assert len(all_indices) == len(set(all_indices)), "Duplicate indices found"


def test_epsilon_smoothing():
    """Test that epsilon smoothing prevents empty allocations."""
    # Create highly imbalanced scenario
    n_samples = 100
    n_clients = 10
    labels = np.array([0] * 95 + [1] * 5)  # Very imbalanced

    # Perform split with low alpha (high heterogeneity)
    client_indices = dirichlet_split_by_class(
        labels=labels,
        num_clients=n_clients,
        alpha=0.01,  # Very low alpha
        seed=42,
        epsilon=1e-6
    )

    # Check that no client is completely empty
    for i, indices in enumerate(client_indices):
        assert len(indices) > 0, f"Client {i} has no samples"


def test_min_sample_enforcement():
    """Test that minimum sample requirement (≥5) is enforced."""
    # Create scenario with enough samples
    n_samples = 200
    n_classes = 5
    n_clients = 10
    labels = np.random.randint(0, n_classes, n_samples)

    # Perform split
    client_indices = dirichlet_split_by_class(
        labels=labels,
        num_clients=n_clients,
        alpha=0.1,
        seed=42
    )

    # Check minimum samples per client
    for i, indices in enumerate(client_indices):
        if len(indices) < 5:
            # Allow small violations if total samples are limited
            assert n_samples / n_clients < 10, f"Client {i} has only {len(indices)} samples"


def test_build_domain_clients():
    """Test domain client building with offloading."""
    # Create dummy index
    index = {
        'samples': [
            {'domain': 'clipart', 'label': i % 10, 'path': f'clipart/img_{i}.jpg'}
            for i in range(500)
        ],
        'domains': ['clipart'],
        'num_classes': 10
    }

    # Build clients
    result = build_domain_clients(
        index=index,
        domain='clipart',
        num_clients=5,
        alpha=0.5,
        unload_ratio=0.2,
        val_ratio=0.1,
        seed=42
    )

    # Check structure
    assert 'clients' in result
    assert 'dc_unload_pool' in result
    assert len(result['clients']) == 5

    # Check each client
    total_local = 0
    total_unload = 0
    total_val = 0

    for client_id, client_data in result['clients'].items():
        assert 'local' in client_data
        assert 'unload' in client_data
        assert 'val' in client_data
        assert 'domain' in client_data

        total_local += len(client_data['local'])
        total_unload += len(client_data['unload'])
        total_val += len(client_data['val'])

        # Check no overlap between sets
        sets = [
            set(client_data['local']),
            set(client_data['unload']),
            set(client_data['val'])
        ]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert len(sets[i] & sets[j]) == 0, "Overlap between data splits"

    # Check totals
    assert total_local + total_unload + total_val == 500
    assert len(result['dc_unload_pool']) == total_unload

    # Check offload ratio approximately correct
    train_total = total_local + total_unload
    actual_ratio = total_unload / train_total
    assert abs(actual_ratio - 0.2) < 0.05, f"Offload ratio {actual_ratio} far from 0.2"