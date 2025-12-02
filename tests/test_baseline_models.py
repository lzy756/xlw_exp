"""Unit tests for baseline model components."""

import pytest
import torch
import torch.nn as nn
from baseline.models.resnet18_single import ResNet18Single


def test_resnet18_single_instantiation():
    """Test that ResNet18Single can be instantiated."""
    num_classes = 126

    model = ResNet18Single(
        num_classes=num_classes,
        pretrained=False
    )

    assert model is not None, "Model should be instantiated"
    assert model.num_classes == num_classes
    assert hasattr(model, 'fc'), "Model should have fc layer"
    assert model.fc.out_features == num_classes

    print(f"✓ ResNet18Single instantiated with {num_classes} classes")


def test_resnet18_single_feature_dim():
    """Test that ResNet18Single outputs 512-d features."""
    num_classes = 10

    model = ResNet18Single(
        num_classes=num_classes,
        pretrained=False
    )

    assert hasattr(model, 'feature_dim'), "Model should have feature_dim property"
    assert model.feature_dim == 512, f"Expected 512-d features, got {model.feature_dim}"

    # Test forward_features output
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        features = model.forward_features(x)

    expected_shape = (batch_size, 512)
    assert features.shape == expected_shape, \
        f"Expected feature shape {expected_shape}, got {features.shape}"

    print(f"✓ ResNet18Single produces 512-d features correctly")


def test_resnet18_single_forward():
    """Test forward pass returns correct output shape."""
    num_classes = 126
    batch_size = 4

    model = ResNet18Single(
        num_classes=num_classes,
        pretrained=False
    )

    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    expected_shape = (batch_size, num_classes)
    assert output.shape == expected_shape, \
        f"Expected output shape {expected_shape}, got {output.shape}"

    print(f"✓ Forward pass produces {output.shape} output")


def test_resnet18_single_forward_with_domain():
    """Test forward pass works with domain argument (ignored)."""
    num_classes = 10
    batch_size = 2

    model = ResNet18Single(
        num_classes=num_classes,
        pretrained=False
    )

    x = torch.randn(batch_size, 3, 224, 224)

    # Test with domain argument (should be ignored)
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x, domain="real")
        output3 = model(x, domain="clipart")

    # All outputs should be the same (domain is ignored)
    assert torch.allclose(output1, output2), "Output should be same regardless of domain"
    assert torch.allclose(output1, output3), "Output should be same regardless of domain"

    print(f"✓ Forward pass ignores domain argument correctly")


def test_state_dict_global():
    """Test state_dict_global exports all parameters."""
    num_classes = 10

    model = ResNet18Single(
        num_classes=num_classes,
        pretrained=False
    )

    # Get global state dict
    state_dict = model.state_dict_global()

    # Verify it contains all parameters
    model_state = model.state_dict()
    assert len(state_dict) == len(model_state), \
        f"state_dict_global should have same size as state_dict"

    for key in model_state:
        assert key in state_dict, f"Key {key} should be in state_dict_global"
        assert torch.allclose(state_dict[key].cpu(), model_state[key].cpu()), \
            f"Values for {key} should match"

    print(f"✓ state_dict_global exports all {len(state_dict)} parameters")


def test_load_state_dict_global():
    """Test load_state_dict_global loads parameters correctly."""
    num_classes = 10

    model1 = ResNet18Single(num_classes=num_classes, pretrained=False)
    model2 = ResNet18Single(num_classes=num_classes, pretrained=False)

    # Models should have different random weights initially
    state1 = model1.state_dict_global()
    state2 = model2.state_dict_global()

    # Load model1's state into model2
    model2.load_state_dict_global(state1)

    # Now model2's state should match model1's
    state2_after = model2.state_dict_global()

    for key in state1:
        assert torch.allclose(state1[key].cpu(), state2_after[key].cpu()), \
            f"Values for {key} should match after load"

    print(f"✓ load_state_dict_global loads parameters correctly")


def test_parameters_all():
    """Test parameters_all returns all trainable parameters."""
    num_classes = 10

    model = ResNet18Single(
        num_classes=num_classes,
        pretrained=False
    )

    all_params = model.parameters_all()
    named_params = dict(model.named_parameters())

    assert len(all_params) == len(named_params), \
        f"parameters_all should return {len(named_params)} parameters"

    # Check that all parameters are included
    param_ids = {id(p) for p in all_params}
    for name, param in named_params.items():
        assert id(param) in param_ids, f"Parameter {name} should be in parameters_all"

    print(f"✓ parameters_all returns all {len(all_params)} parameters")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    num_classes = 10

    model = ResNet18Single(
        num_classes=num_classes,
        pretrained=False
    )

    # Create dummy input and target
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    target = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, target)

    # Backward pass
    loss.backward()

    # Check that parameters have gradients
    params_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None and torch.abs(param.grad).sum() > 0:
            params_with_grad += 1

    assert params_with_grad > 0, "At least some parameters should have gradients"

    print(f"✓ Gradients flow through model ({params_with_grad} params with non-zero grads)")


def test_no_lora_blocks():
    """Test that ResNet18Single has no LoRA blocks."""
    num_classes = 10

    model = ResNet18Single(
        num_classes=num_classes,
        pretrained=False
    )

    # Should not have lora_blocks attribute
    assert not hasattr(model, 'lora_blocks'), \
        "ResNet18Single should not have lora_blocks"

    # Check that no parameters contain 'lora' in name
    for name, param in model.named_parameters():
        assert 'lora' not in name.lower(), \
            f"Parameter {name} should not be LoRA-related"

    print(f"✓ ResNet18Single has no LoRA components")


def test_single_head():
    """Test that ResNet18Single has only one head."""
    num_classes = 10

    model = ResNet18Single(
        num_classes=num_classes,
        pretrained=False
    )

    # Should have single fc layer, not heads dict
    assert hasattr(model, 'fc'), "Model should have single fc layer"
    assert not hasattr(model, 'heads'), \
        "ResNet18Single should not have domain-specific heads"

    # Check fc layer properties
    assert isinstance(model.fc, nn.Linear), "fc should be Linear layer"
    assert model.fc.in_features == 512, "fc should accept 512-d features"
    assert model.fc.out_features == num_classes, \
        f"fc should output {num_classes} classes"

    print(f"✓ ResNet18Single has single head (512→{num_classes})")


if __name__ == '__main__':
    test_resnet18_single_instantiation()
    test_resnet18_single_feature_dim()
    test_resnet18_single_forward()
    test_resnet18_single_forward_with_domain()
    test_state_dict_global()
    test_load_state_dict_global()
    test_parameters_all()
    test_gradient_flow()
    test_no_lora_blocks()
    test_single_head()

    print("\nAll baseline model tests passed!")
