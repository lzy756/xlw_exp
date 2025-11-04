"""Unit tests for LocalTrainer."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from core.trainer import LocalTrainer
from models.resnet18_lora import ResNet18_EAPH


def test_train_client_updates_parameters():
    """Test that train_client() updates model parameters."""
    # Setup
    model = ResNet18_EAPH(
        num_classes=10,
        domains=['clipart', 'real'],
        lora_rank=4,
        pretrained=False
    )
    trainer = LocalTrainer(
        model=model,
        device='cpu',
        lr_theta=0.01,
        lr_phi=0.01,
        weight_decay=1e-4,
        cosine_lr=False
    )

    # Create dummy dataset
    x = torch.randn(32, 3, 224, 224)
    y = torch.randint(0, 10, (32,))
    domains = torch.zeros(32, dtype=torch.long)  # domain placeholder
    dataset = TensorDataset(x, y, domains)

    # Get initial parameters
    theta_before = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if 'heads.' not in k and 'lora_blocks.' not in k
    }
    phi_before = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if 'lora_blocks.' in k or 'heads.clipart' in k
    }

    # Train
    theta_state, phi_state, acc = trainer.train_client(
        domain='clipart',
        dataset=dataset,
        batch_size=8,
        local_steps=1
    )

    # Verify parameters changed
    assert len(theta_state) > 0, "Theta state should not be empty"
    assert len(phi_state) > 0, "Phi state should not be empty"
    assert 0 <= acc <= 100, f"Accuracy should be in [0, 100], got {acc}"

    # Verify theta parameters were updated
    model.load_state_dict(theta_state, strict=False)
    theta_after = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if 'heads.' not in k and 'lora_blocks.' not in k
    }

    # Check that at least some theta parameters changed
    changed = False
    for key in theta_before:
        if key in theta_after:
            if not torch.allclose(theta_before[key], theta_after[key]):
                changed = True
                break
    assert changed, "Theta parameters should have changed after training"

    print(f"Test passed! Train accuracy: {acc:.2f}%")


def test_theta_phi_separation():
    """Test that theta and phi parameters are correctly separated."""
    model = ResNet18_EAPH(
        num_classes=10,
        domains=['clipart', 'real'],
        lora_rank=4,
        pretrained=False
    )

    # Get theta and phi parameters
    theta_params = set(id(p) for p in model.parameters_theta())
    phi_clipart = set(id(p) for p in model.parameters_phi('clipart'))
    phi_real = set(id(p) for p in model.parameters_phi('real'))

    # Verify no overlap between theta and phi
    overlap_clipart = theta_params & phi_clipart
    overlap_real = theta_params & phi_real
    assert len(overlap_clipart) == 0, "Theta and phi (clipart) parameters should not overlap"
    assert len(overlap_real) == 0, "Theta and phi (real) parameters should not overlap"

    # Verify LoRA parameters are shared between domains
    lora_overlap = phi_clipart & phi_real
    assert len(lora_overlap) > 0, "LoRA parameters should be shared between domains"

    # Verify domain heads are separate
    phi_clipart_only = phi_clipart - phi_real
    phi_real_only = phi_real - phi_clipart
    assert len(phi_clipart_only) > 0, "Should have clipart-specific parameters (head)"
    assert len(phi_real_only) > 0, "Should have real-specific parameters (head)"

    # Verify coverage (all parameters should be either theta or phi)
    all_params = set(id(p) for p in model.parameters())

    # Each parameter should be in theta OR in at least one domain's phi
    for param_id in all_params:
        in_theta = param_id in theta_params
        in_phi = param_id in (phi_clipart | phi_real)
        assert in_theta or in_phi, "All parameters should be in theta or phi"

    print("Parameter separation test passed!")
    print(f"  Theta params: {len(theta_params)}")
    print(f"  Phi clipart params: {len(phi_clipart)}")
    print(f"  Phi real params: {len(phi_real)}")
    print(f"  Shared LoRA params: {len(lora_overlap)}")


def test_evaluate_domain():
    """Test that evaluate_domain() returns correct format."""
    model = ResNet18_EAPH(
        num_classes=10,
        domains=['clipart', 'real'],
        lora_rank=4,
        pretrained=False
    )
    trainer = LocalTrainer(
        model=model,
        device='cpu',
        lr_theta=0.01,
        lr_phi=0.01,
        cosine_lr=False
    )

    # Create dummy validation dataset
    x = torch.randn(16, 3, 224, 224)
    y = torch.randint(0, 10, (16,))
    domains = torch.zeros(16, dtype=torch.long)
    dataset = TensorDataset(x, y, domains)

    # Evaluate
    val_loss, val_acc, L_e = trainer.evaluate_domain(
        domain='clipart',
        dataset=dataset,
        batch_size=8,
        edge_manager=None
    )

    # Verify return types and ranges
    assert isinstance(val_loss, float), "val_loss should be float"
    assert isinstance(val_acc, float), "val_acc should be float"
    assert isinstance(L_e, float), "L_e should be float"
    assert val_loss >= 0, "val_loss should be non-negative"
    assert 0 <= val_acc <= 100, f"val_acc should be in [0, 100], got {val_acc}"
    assert L_e >= 0, "L_e should be non-negative"

    # Without edge_manager, L_e should equal val_loss
    assert L_e == val_loss, "L_e should equal val_loss when edge_manager is None"

    print(f"Evaluation test passed! val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%, L_e: {L_e:.4f}")


def test_train_client_different_lr_phi():
    """Test that train_client() accepts custom lr_phi."""
    model = ResNet18_EAPH(
        num_classes=10,
        domains=['clipart', 'real'],
        lora_rank=4,
        pretrained=False
    )
    trainer = LocalTrainer(
        model=model,
        device='cpu',
        lr_theta=0.01,
        lr_phi=0.01,
        cosine_lr=False
    )

    # Create dummy dataset
    x = torch.randn(16, 3, 224, 224)
    y = torch.randint(0, 10, (16,))
    domains = torch.zeros(16, dtype=torch.long)
    dataset = TensorDataset(x, y, domains)

    # Train with custom lr_phi
    theta_state, phi_state, acc = trainer.train_client(
        domain='clipart',
        dataset=dataset,
        batch_size=8,
        local_steps=1,
        lr_phi=0.05  # Custom learning rate
    )

    # Should complete without error
    assert len(theta_state) > 0
    assert len(phi_state) > 0
    assert 0 <= acc <= 100

    print(f"Custom lr_phi test passed! Accuracy: {acc:.2f}%")


if __name__ == '__main__':
    test_train_client_updates_parameters()
    test_theta_phi_separation()
    test_evaluate_domain()
    test_train_client_different_lr_phi()
    print("\nAll trainer tests passed!")
