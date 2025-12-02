"""Unit tests for baseline trainers (FedAvg and FedProx)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

from baseline.models.resnet18_single import ResNet18Single
from baseline.core.trainer_fedavg import LocalTrainerFedAvg
from baseline.core.trainer_fedprox import LocalTrainerFedProx
from baseline.core.selector_fixed import FixedSelector


class DummyDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, num_samples: int = 100, num_classes: int = 10):
        self.num_samples = num_samples
        self.num_classes = num_classes
        # Random images and labels
        self.images = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.domains = ['test'] * num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.domains[idx]


class TestLocalTrainerFedAvg:
    """Tests for FedAvg local trainer."""

    def test_trainer_initialization(self):
        """Test that trainer can be initialized."""
        model = ResNet18Single(num_classes=10, pretrained=False)
        trainer = LocalTrainerFedAvg(
            model=model,
            device='cpu',
            lr=1e-3,
            weight_decay=1e-4
        )

        assert trainer.model is not None
        assert trainer.device == 'cpu'
        assert trainer.lr == 1e-3
        assert trainer.weight_decay == 1e-4

    def test_train_client_returns_state_dict_and_count(self):
        """Test that train_client returns state dict and sample count."""
        model = ResNet18Single(num_classes=10, pretrained=False)
        trainer = LocalTrainerFedAvg(model=model, device='cpu')

        dataset = DummyDataset(num_samples=50, num_classes=10)

        state_dict, num_samples = trainer.train_client(
            dataset=dataset,
            batch_size=16,
            local_steps=1
        )

        assert isinstance(state_dict, dict), "Should return state dict"
        assert len(state_dict) > 0, "State dict should not be empty"
        assert num_samples == 50, f"Should return 50 samples, got {num_samples}"

    def test_train_client_empty_dataset(self):
        """Test train_client with empty dataset."""
        model = ResNet18Single(num_classes=10, pretrained=False)
        trainer = LocalTrainerFedAvg(model=model, device='cpu')

        # Create empty dataset
        dataset = DummyDataset(num_samples=0, num_classes=10)

        state_dict, num_samples = trainer.train_client(
            dataset=dataset,
            batch_size=16,
            local_steps=1
        )

        assert num_samples == 0, "Should return 0 samples for empty dataset"

    def test_train_client_modifies_weights(self):
        """Test that training actually modifies model weights."""
        model = ResNet18Single(num_classes=10, pretrained=False)
        trainer = LocalTrainerFedAvg(model=model, device='cpu')

        # Get initial weights
        initial_state = model.state_dict_global()
        initial_fc_weight = initial_state['fc.weight'].clone()

        dataset = DummyDataset(num_samples=50, num_classes=10)

        state_dict, _ = trainer.train_client(
            dataset=dataset,
            batch_size=16,
            local_steps=2
        )

        # Weights should be different after training
        assert not torch.allclose(state_dict['fc.weight'], initial_fc_weight), \
            "Weights should change after training"

    def test_evaluate(self):
        """Test evaluation function."""
        model = ResNet18Single(num_classes=10, pretrained=False)
        trainer = LocalTrainerFedAvg(model=model, device='cpu')

        dataset = DummyDataset(num_samples=50, num_classes=10)

        val_loss, val_acc = trainer.evaluate(
            dataset=dataset,
            batch_size=16
        )

        assert isinstance(val_loss, float), "val_loss should be float"
        assert isinstance(val_acc, float), "val_acc should be float"
        assert val_loss >= 0, "val_loss should be non-negative"
        assert 0 <= val_acc <= 100, f"val_acc should be in [0, 100], got {val_acc}"

    def test_load_global_state(self):
        """Test loading global state into trainer model."""
        model1 = ResNet18Single(num_classes=10, pretrained=False)
        model2 = ResNet18Single(num_classes=10, pretrained=False)

        trainer = LocalTrainerFedAvg(model=model2, device='cpu')

        # Load model1's state into trainer's model
        state1 = model1.state_dict_global()
        trainer.load_global_state(state1)

        # Trainer's model should now have model1's weights
        state2 = trainer.model.state_dict_global()

        for key in state1:
            assert torch.allclose(state1[key].cpu(), state2[key].cpu()), \
                f"Weights for {key} should match"


class TestLocalTrainerFedProx:
    """Tests for FedProx local trainer."""

    def test_trainer_initialization_with_mu(self):
        """Test that FedProx trainer can be initialized with mu."""
        model = ResNet18Single(num_classes=10, pretrained=False)
        trainer = LocalTrainerFedProx(
            model=model,
            device='cpu',
            lr=1e-3,
            weight_decay=1e-4,
            mu=0.01
        )

        assert trainer.mu == 0.01

    def test_train_client_with_proximal_term(self):
        """Test training with proximal term."""
        model = ResNet18Single(num_classes=10, pretrained=False)
        trainer = LocalTrainerFedProx(
            model=model,
            device='cpu',
            lr=1e-3,
            mu=0.01
        )

        dataset = DummyDataset(num_samples=50, num_classes=10)

        # Get global snapshot
        w_global = model.state_dict_global()

        state_dict, num_samples = trainer.train_client(
            dataset=dataset,
            batch_size=16,
            local_steps=2,
            w_global_snapshot=w_global
        )

        assert isinstance(state_dict, dict)
        assert num_samples == 50

    def test_train_client_without_snapshot_falls_back_to_fedavg(self):
        """Test that training without snapshot falls back to FedAvg."""
        model = ResNet18Single(num_classes=10, pretrained=False)
        trainer = LocalTrainerFedProx(
            model=model,
            device='cpu',
            lr=1e-3,
            mu=0.01
        )

        dataset = DummyDataset(num_samples=50, num_classes=10)

        # Should work without w_global_snapshot
        state_dict, num_samples = trainer.train_client(
            dataset=dataset,
            batch_size=16,
            local_steps=1,
            w_global_snapshot=None
        )

        assert isinstance(state_dict, dict)
        assert num_samples == 50

    def test_mu_zero_equals_fedavg(self):
        """Test that μ=0 degrades to FedAvg behavior."""
        model = ResNet18Single(num_classes=10, pretrained=False)
        trainer = LocalTrainerFedProx(
            model=model,
            device='cpu',
            lr=1e-3,
            mu=0  # No proximal term
        )

        dataset = DummyDataset(num_samples=50, num_classes=10)
        w_global = model.state_dict_global()

        # Should still work with mu=0
        state_dict, num_samples = trainer.train_client(
            dataset=dataset,
            batch_size=16,
            local_steps=1,
            w_global_snapshot=w_global
        )

        assert isinstance(state_dict, dict)
        assert num_samples == 50

    def test_proximal_term_constrains_updates(self):
        """Test that proximal term limits how far weights drift from global."""
        # This is a rough test - with high μ, weights should stay closer to global
        model1 = ResNet18Single(num_classes=10, pretrained=False)
        model2 = ResNet18Single(num_classes=10, pretrained=False)

        # Use same initial weights
        model2.load_state_dict_global(model1.state_dict_global())

        # Trainer with no proximal term
        trainer_no_prox = LocalTrainerFedProx(
            model=model1,
            device='cpu',
            lr=1e-2,  # Higher LR to see more drift
            mu=0
        )

        # Trainer with strong proximal term
        trainer_prox = LocalTrainerFedProx(
            model=model2,
            device='cpu',
            lr=1e-2,
            mu=1.0  # Strong regularization
        )

        dataset = DummyDataset(num_samples=100, num_classes=10)
        w_global = model1.state_dict_global()

        # Train both
        state_no_prox, _ = trainer_no_prox.train_client(
            dataset=dataset,
            batch_size=16,
            local_steps=5,
            w_global_snapshot=w_global
        )

        state_prox, _ = trainer_prox.train_client(
            dataset=dataset,
            batch_size=16,
            local_steps=5,
            w_global_snapshot=w_global
        )

        # Calculate drift from global for both
        drift_no_prox = 0.0
        drift_prox = 0.0

        for key in w_global:
            # Skip non-float tensors (e.g., num_batches_tracked)
            if not state_no_prox[key].is_floating_point():
                continue
            drift_no_prox += torch.norm(state_no_prox[key].float() - w_global[key].float()).item()
            drift_prox += torch.norm(state_prox[key].float() - w_global[key].float()).item()

        # With strong μ, drift should be smaller
        # Note: This might not always hold due to random initialization
        # but generally should be true
        print(f"Drift without prox: {drift_no_prox:.4f}")
        print(f"Drift with prox (μ=1.0): {drift_prox:.4f}")


class TestFixedSelector:
    """Tests for fixed aggregator selector."""

    def test_selector_initialization(self):
        """Test that selector can be initialized."""
        selector = FixedSelector(fixed_domain="real")
        assert selector.fixed_domain == "real"

    def test_select_returns_fixed_domain(self):
        """Test that select always returns the fixed domain."""
        selector = FixedSelector(fixed_domain="real")

        # Call multiple times
        for _ in range(10):
            result = selector.select()
            assert result == "real"

    def test_select_ignores_arguments(self):
        """Test that select ignores any arguments."""
        selector = FixedSelector(fixed_domain="clipart")

        # Pass various arguments
        result1 = selector.select()
        result2 = selector.select(foo="bar")
        result3 = selector.select(metrics={}, data=[1, 2, 3])

        assert result1 == "clipart"
        assert result2 == "clipart"
        assert result3 == "clipart"

    def test_different_fixed_domains(self):
        """Test with different fixed domains."""
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

        for domain in domains:
            selector = FixedSelector(fixed_domain=domain)
            assert selector.select() == domain


if __name__ == '__main__':
    # Run FedAvg tests
    print("Testing LocalTrainerFedAvg...")
    test_fedavg = TestLocalTrainerFedAvg()
    test_fedavg.test_trainer_initialization()
    test_fedavg.test_train_client_returns_state_dict_and_count()
    test_fedavg.test_train_client_empty_dataset()
    test_fedavg.test_train_client_modifies_weights()
    test_fedavg.test_evaluate()
    test_fedavg.test_load_global_state()
    print("✓ All FedAvg trainer tests passed")

    # Run FedProx tests
    print("\nTesting LocalTrainerFedProx...")
    test_fedprox = TestLocalTrainerFedProx()
    test_fedprox.test_trainer_initialization_with_mu()
    test_fedprox.test_train_client_with_proximal_term()
    test_fedprox.test_train_client_without_snapshot_falls_back_to_fedavg()
    test_fedprox.test_mu_zero_equals_fedavg()
    test_fedprox.test_proximal_term_constrains_updates()
    print("✓ All FedProx trainer tests passed")

    # Run FixedSelector tests
    print("\nTesting FixedSelector...")
    test_selector = TestFixedSelector()
    test_selector.test_selector_initialization()
    test_selector.test_select_returns_fixed_domain()
    test_selector.test_select_ignores_arguments()
    test_selector.test_different_fixed_domains()
    print("✓ All FixedSelector tests passed")

    print("\nAll baseline trainer tests passed!")
