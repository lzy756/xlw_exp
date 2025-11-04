"""Unit tests for EdgeManager."""

import pytest
import torch
import torch.nn as nn
from core.edge_manager import EdgeManager


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.fc(x)


def test_get_set_phi_roundtrip():
    """Test that get_phi and set_phi work correctly for roundtrip storage."""
    model = DummyModel()
    domains = ['domain1', 'domain2']

    edge_manager = EdgeManager(
        model=model,
        domains=domains,
        num_classes=10,
        proj_dim=64,
        device='cpu'
    )

    # Initially, phi should be None
    assert edge_manager.get_phi('domain1') is None
    assert edge_manager.get_phi('domain2') is None

    # Create a dummy state dict
    dummy_state = {
        'lora_blocks.0.weight': torch.randn(3, 3),
        'heads.domain1.weight': torch.randn(10, 512)
    }

    # Set phi for domain1
    edge_manager.set_phi('domain1', dummy_state)

    # Get phi back
    retrieved_state = edge_manager.get_phi('domain1')

    # Verify it's the same
    assert retrieved_state is not None
    assert 'lora_blocks.0.weight' in retrieved_state
    assert 'heads.domain1.weight' in retrieved_state
    assert torch.equal(retrieved_state['lora_blocks.0.weight'], dummy_state['lora_blocks.0.weight'])
    assert torch.equal(retrieved_state['heads.domain1.weight'], dummy_state['heads.domain1.weight'])

    # domain2 should still be None
    assert edge_manager.get_phi('domain2') is None


def test_begin_round_increments_coverage():
    """Test that begin_round increments coverage counters for all domains."""
    model = DummyModel()
    domains = ['domain1', 'domain2', 'domain3']

    edge_manager = EdgeManager(
        model=model,
        domains=domains,
        num_classes=10,
        proj_dim=64,
        device='cpu'
    )

    # Initially, all coverage should be 0
    for domain in domains:
        assert edge_manager.coverage[domain] == 0.0

    # Call begin_round once
    edge_manager.begin_round()

    # All coverage should be 1
    for domain in domains:
        assert edge_manager.coverage[domain] == 1.0

    # Call begin_round again
    edge_manager.begin_round()

    # All coverage should be 2
    for domain in domains:
        assert edge_manager.coverage[domain] == 2.0


def test_compute_drift_returns_zero_first_call():
    """Test that compute_drift returns 0 on first call (no previous prototype)."""
    model = DummyModel()
    domains = ['domain1']

    edge_manager = EdgeManager(
        model=model,
        domains=domains,
        num_classes=10,
        proj_dim=64,
        device='cpu'
    )

    # Simulate some prototype accumulation
    # Add some dummy data to proto_sum and proto_cnt
    edge_manager.proto_sum['domain1'][0] = torch.randn(64)
    edge_manager.proto_cnt['domain1'][0] = 10.0
    edge_manager.proto_sum['domain1'][1] = torch.randn(64)
    edge_manager.proto_cnt['domain1'][1] = 5.0

    # First call should return 0 (no previous prototype)
    drift = edge_manager.compute_drift('domain1')
    assert drift == 0.0

    # prev_proto should now be set
    assert edge_manager.prev_proto['domain1'] is not None

    # Accumulators should be reset
    assert torch.all(edge_manager.proto_sum['domain1'] == 0)
    assert torch.all(edge_manager.proto_cnt['domain1'] == 0)


def test_random_projection_deterministic():
    """Test that random projection matrix is deterministic with seed=0."""
    model1 = DummyModel()
    model2 = DummyModel()
    domains = ['domain1']

    edge_manager1 = EdgeManager(
        model=model1,
        domains=domains,
        num_classes=10,
        proj_dim=64,
        device='cpu'
    )

    edge_manager2 = EdgeManager(
        model=model2,
        domains=domains,
        num_classes=10,
        proj_dim=64,
        device='cpu'
    )

    # Projection matrices should be identical
    assert torch.equal(edge_manager1.proj, edge_manager2.proj)


def test_drift_weighted_l2_correct():
    """Test that drift computation uses weighted L2 distance."""
    model = DummyModel()
    domains = ['domain1']

    edge_manager = EdgeManager(
        model=model,
        domains=domains,
        num_classes=3,  # Small number for easy testing
        proj_dim=4,
        device='cpu'
    )

    # Set up first round prototypes
    # Class 0: 10 samples
    edge_manager.proto_sum['domain1'][0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    edge_manager.proto_cnt['domain1'][0] = 10.0
    # Class 1: 20 samples
    edge_manager.proto_sum['domain1'][1] = torch.tensor([0.0, 2.0, 0.0, 0.0])
    edge_manager.proto_cnt['domain1'][1] = 20.0
    # Class 2: 10 samples
    edge_manager.proto_sum['domain1'][2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    edge_manager.proto_cnt['domain1'][2] = 10.0

    # First call returns 0
    drift1 = edge_manager.compute_drift('domain1')
    assert drift1 == 0.0

    # Set up second round prototypes (shifted)
    # Class 0: 10 samples (shifted by 0.1 in first dimension)
    edge_manager.proto_sum['domain1'][0] = torch.tensor([1.1, 0.0, 0.0, 0.0])
    edge_manager.proto_cnt['domain1'][0] = 10.0
    # Class 1: 20 samples (no change)
    edge_manager.proto_sum['domain1'][1] = torch.tensor([0.0, 2.0, 0.0, 0.0])
    edge_manager.proto_cnt['domain1'][1] = 20.0
    # Class 2: 10 samples (shifted by 0.2 in third dimension)
    edge_manager.proto_sum['domain1'][2] = torch.tensor([0.0, 0.0, 1.2, 0.0])
    edge_manager.proto_cnt['domain1'][2] = 10.0

    # Second call should compute drift
    drift2 = edge_manager.compute_drift('domain1')

    # Drift should be > 0
    assert drift2 > 0.0

    # Manual calculation:
    # Prototypes (after division by counts):
    # Round 1: proto[0] = [0.1, 0, 0, 0], proto[1] = [0, 0.1, 0, 0], proto[2] = [0, 0, 0.1, 0]
    # Round 2: proto[0] = [0.11, 0, 0, 0], proto[1] = [0, 0.1, 0, 0], proto[2] = [0, 0, 0.12, 0]
    # L2 norms: diff[0] = 0.01, diff[1] = 0, diff[2] = 0.02
    # Weights: pi[0] = 10/40 = 0.25, pi[1] = 20/40 = 0.5, pi[2] = 10/40 = 0.25
    # Drift = 0.25 * 0.01 + 0.5 * 0 + 0.25 * 0.02 = 0.0025 + 0 + 0.005 = 0.0075
    # (approximately, accounting for floating point errors)
    assert abs(drift2 - 0.0075) < 0.001


def test_drift_resets_accumulators():
    """Test that compute_drift resets accumulators after computing drift."""
    model = DummyModel()
    domains = ['domain1']

    edge_manager = EdgeManager(
        model=model,
        domains=domains,
        num_classes=10,
        proj_dim=64,
        device='cpu'
    )

    # Add some data
    edge_manager.proto_sum['domain1'][0] = torch.randn(64)
    edge_manager.proto_cnt['domain1'][0] = 10.0
    edge_manager.proto_sum['domain1'][5] = torch.randn(64)
    edge_manager.proto_cnt['domain1'][5] = 20.0

    # Compute drift
    edge_manager.compute_drift('domain1')

    # Accumulators should be zero
    assert torch.all(edge_manager.proto_sum['domain1'] == 0)
    assert torch.all(edge_manager.proto_cnt['domain1'] == 0)


def test_proto_accumulation_correct():
    """Test that update_eval_stats correctly accumulates prototypes."""
    model = DummyModel()
    domains = ['domain1']

    edge_manager = EdgeManager(
        model=model,
        domains=domains,
        num_classes=3,
        proj_dim=4,
        device='cpu'
    )

    # Create dummy features and labels
    # 5 samples: 2 of class 0, 2 of class 1, 1 of class 2
    feats = torch.randn(5, 512)
    labels = torch.tensor([0, 0, 1, 1, 2])

    # Update stats
    L_e = edge_manager.update_eval_stats(
        domain='domain1',
        val_loss=0.5,
        feats=feats,
        labels=labels,
        tau=1.0
    )

    # Check that prototypes were accumulated
    assert edge_manager.proto_cnt['domain1'][0] == 2.0
    assert edge_manager.proto_cnt['domain1'][1] == 2.0
    assert edge_manager.proto_cnt['domain1'][2] == 1.0

    # proto_sum should be non-zero for these classes
    assert not torch.all(edge_manager.proto_sum['domain1'][0] == 0)
    assert not torch.all(edge_manager.proto_sum['domain1'][1] == 0)
    assert not torch.all(edge_manager.proto_sum['domain1'][2] == 0)

    # L_e should be returned
    assert L_e == 0.5


def test_end_round_with_aggregator():
    """Test that end_round_with_aggregator resets coverage and updates stability."""
    model = DummyModel()
    domains = ['domain1', 'domain2', 'domain3']

    edge_manager = EdgeManager(
        model=model,
        domains=domains,
        num_classes=10,
        proj_dim=64,
        device='cpu'
    )

    # Simulate a few rounds
    edge_manager.begin_round()
    edge_manager.begin_round()
    edge_manager.begin_round()

    # All coverage should be 3
    for domain in domains:
        assert edge_manager.coverage[domain] == 3.0

    # Select domain1 as aggregator
    edge_manager.end_round_with_aggregator('domain1')

    # domain1 coverage should be reset to 0
    assert edge_manager.coverage['domain1'] == 0.0
    # Others should still be 3
    assert edge_manager.coverage['domain2'] == 3.0
    assert edge_manager.coverage['domain3'] == 3.0

    # No stability bonus yet (first time selecting domain1)
    assert edge_manager.stay_bonus['domain1'] == 0.0
    assert edge_manager.stay_bonus['domain2'] == 0.0
    assert edge_manager.stay_bonus['domain3'] == 0.0

    # Select domain1 again
    edge_manager.end_round_with_aggregator('domain1')

    # domain1 should get stability bonus
    assert edge_manager.stay_bonus['domain1'] == 0.2
    assert edge_manager.stay_bonus['domain2'] == 0.0
    assert edge_manager.stay_bonus['domain3'] == 0.0

    # Select domain2
    edge_manager.end_round_with_aggregator('domain2')

    # domain1 loses stability bonus
    assert edge_manager.stay_bonus['domain1'] == 0.0
    # domain2 doesn't get it (not selected twice in a row)
    assert edge_manager.stay_bonus['domain2'] == 0.0
    assert edge_manager.stay_bonus['domain3'] == 0.0
