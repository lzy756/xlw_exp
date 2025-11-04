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


def test_offload_ratio_enforcement():
    """Test that offload ratio is enforced correctly for different ρ values."""
    # Create dummy index
    index = {
        'samples': [
            {'domain': 'real', 'label': i % 10, 'path': f'real/img_{i}.jpg'}
            for i in range(1000)
        ],
        'domains': ['real'],
        'num_classes': 10
    }

    # Test different offload ratios
    for unload_ratio in [0.0, 0.1, 0.2, 0.4]:
        result = build_domain_clients(
            index=index,
            domain='real',
            num_clients=10,
            alpha=0.5,
            unload_ratio=unload_ratio,
            val_ratio=0.1,
            seed=42
        )

        # Calculate actual offload ratio
        total_local = sum(len(client['local']) for client in result['clients'].values())
        total_unload = sum(len(client['unload']) for client in result['clients'].values())
        total_dc_pool = len(result['dc_unload_pool'])

        # DC pool should equal sum of client unloads
        assert total_dc_pool == total_unload, \
            f"DC pool size {total_dc_pool} != sum of unloads {total_unload}"

        # Check ratio
        if total_local + total_unload > 0:
            actual_ratio = total_unload / (total_local + total_unload)
            assert abs(actual_ratio - unload_ratio) < 0.05, \
                f"Ratio ρ={unload_ratio}: expected {unload_ratio}, got {actual_ratio}"

        # If ratio is 0, no offloading should occur
        if unload_ratio == 0.0:
            assert total_unload == 0, "No offloading should occur when ρ=0"
            assert total_dc_pool == 0, "DC pool should be empty when ρ=0"


def test_offload_pool_aggregation():
    """Test that DC offload pool correctly aggregates data from all clients."""
    # Create dummy index
    index = {
        'samples': [
            {'domain': 'sketch', 'label': i % 5, 'path': f'sketch/img_{i}.jpg'}
            for i in range(500)
        ],
        'domains': ['sketch'],
        'num_classes': 5
    }

    result = build_domain_clients(
        index=index,
        domain='sketch',
        num_clients=8,
        alpha=0.3,
        unload_ratio=0.2,
        val_ratio=0.15,
        seed=42
    )

    # Collect all unloaded indices from clients
    client_unload_indices = []
    for client_data in result['clients'].values():
        client_unload_indices.extend(client_data['unload'])

    # DC pool should contain exactly these indices
    dc_pool = result['dc_unload_pool']

    assert len(dc_pool) == len(client_unload_indices), \
        f"DC pool size {len(dc_pool)} != total client unloads {len(client_unload_indices)}"

    # All indices should match
    assert set(dc_pool) == set(client_unload_indices), \
        "DC pool indices don't match client unload indices"

    # Verify offloaded data is excluded from local sets
    for client_data in result['clients'].values():
        local_set = set(client_data['local'])
        unload_set = set(client_data['unload'])

        # No overlap between local and unload
        assert len(local_set & unload_set) == 0, \
            "Client has overlap between local and offload data"

        # All unloaded indices should be in DC pool
        assert unload_set.issubset(set(dc_pool)), \
            "Client offload data not in DC pool"