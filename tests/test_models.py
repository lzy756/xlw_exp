"""Unit tests for model components."""

import pytest
import torch
import torch.nn as nn
from models.resnet18_lora import LoRAConv2d, ResNet18_EAPH
from models.resnet50_lora import ResNet50_EAPH


def test_lora_conv2d_residual_path():
    """Test that LoRAConv2d correctly implements residual connection."""
    in_channels = 16
    out_channels = 32
    kernel_size = 3
    rank = 4
    alpha = 4.0

    # Create LoRA layer
    lora_layer = LoRAConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        rank=rank,
        alpha=alpha,
        padding=1
    )

    # Create input tensor
    batch_size = 2
    height = width = 8
    x = torch.randn(batch_size, in_channels, height, width)

    # Forward pass
    output = lora_layer(x)

    # Check output shape
    expected_shape = (batch_size, out_channels, height, width)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Check that LoRA path exists
    assert hasattr(lora_layer, 'lora_down'), "LoRA layer should have lora_down"
    assert hasattr(lora_layer, 'lora_up'), "LoRA layer should have lora_up"

    # Verify residual connection: output should be base + lora
    # This is implicit in the forward pass, but we can check by disabling LoRA
    with torch.no_grad():
        # Zero out LoRA weights
        lora_layer.lora_down.weight.zero_()
        lora_layer.lora_up.weight.zero_()

        # Output should now be just the base conv
        output_no_lora = lora_layer(x)

        # Reset LoRA weights
        nn.init.kaiming_uniform_(lora_layer.lora_down.weight)
        nn.init.zeros_(lora_layer.lora_up.weight)

        # Output with LoRA should be different
        output_with_lora = lora_layer(x)

        # When LoRA is zeroed, outputs should match base
        # When LoRA is non-zero (initialized), they should differ
        # (Note: we initialized lora_up to zeros, so actually they'll be same)
        # Let's add some values to lora_up
        lora_layer.lora_up.weight.normal_(0, 0.01)
        output_with_lora_nonzero = lora_layer(x)

        # This should be different from the no-LoRA case
        assert not torch.allclose(output_no_lora, output_with_lora_nonzero), \
            "LoRA should affect output when weights are non-zero"


def test_parameters_theta_excludes_lora():
    """Test that parameters_theta excludes LoRA and head parameters."""
    num_classes = 10
    domains = ['domain1', 'domain2']
    lora_rank = 4
    lora_alpha = 4.0

    model = ResNet18_EAPH(
        num_classes=num_classes,
        domains=domains,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        pretrained=False
    )

    # Get theta parameters
    theta_params = list(model.parameters_theta())

    # Get all parameter names
    all_param_names = [name for name, _ in model.named_parameters()]
    theta_param_ids = {id(p) for p in theta_params}

    # Check that no LoRA parameters are included
    for name, param in model.named_parameters():
        if 'lora_blocks' in name or 'heads' in name:
            assert id(param) not in theta_param_ids, \
                f"Parameter {name} should not be in theta parameters"
        else:
            assert id(param) in theta_param_ids, \
                f"Parameter {name} should be in theta parameters"

    # Verify we got some parameters
    assert len(theta_params) > 0, "Theta parameters should not be empty"

    print(f"✓ Theta has {len(theta_params)} parameters (excluding LoRA and heads)")


def test_parameters_phi_includes_lora():
    """Test that parameters_phi includes LoRA and domain-specific head parameters."""
    num_classes = 10
    domains = ['domain1', 'domain2', 'domain3']
    lora_rank = 4
    lora_alpha = 4.0

    model = ResNet18_EAPH(
        num_classes=num_classes,
        domains=domains,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        pretrained=False
    )

    # Test for each domain
    for domain in domains:
        phi_params = list(model.parameters_phi(domain))
        phi_param_ids = {id(p) for p in phi_params}

        # Check that LoRA parameters are included
        lora_found = False
        head_found = False

        for name, param in model.named_parameters():
            if 'lora_blocks' in name:
                assert id(param) in phi_param_ids, \
                    f"LoRA parameter {name} should be in phi parameters for {domain}"
                lora_found = True
            elif f'heads.{domain}' in name:
                assert id(param) in phi_param_ids, \
                    f"Head parameter {name} should be in phi parameters for {domain}"
                head_found = True
            elif 'heads' in name and domain not in name:
                # Other domain heads should NOT be included
                assert id(param) not in phi_param_ids, \
                    f"Other domain head {name} should not be in phi parameters for {domain}"

        assert lora_found, f"LoRA parameters should be in phi for {domain}"
        assert head_found, f"Domain head should be in phi for {domain}"

        print(f"✓ Phi for {domain} has {len(phi_params)} parameters (LoRA + head)")


def test_forward_features():
    """Test that forward_features returns correct feature dimension."""
    num_classes = 10
    domains = ['domain1']
    lora_rank = 4
    lora_alpha = 4.0

    model = ResNet18_EAPH(
        num_classes=num_classes,
        domains=domains,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        pretrained=False
    )

    # Create dummy input (batch of images)
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    # Extract features
    with torch.no_grad():
        features = model.forward_features(x)

    # ResNet18 should output 512-dim features
    expected_shape = (batch_size, 512)
    assert features.shape == expected_shape, \
        f"Expected feature shape {expected_shape}, got {features.shape}"

    print(f"✓ Forward features produces {features.shape} output")


def test_domain_heads_separate():
    """Test that each domain has its own classification head."""
    num_classes = 10
    domains = ['domain1', 'domain2', 'domain3']
    lora_rank = 4
    lora_alpha = 4.0

    model = ResNet18_EAPH(
        num_classes=num_classes,
        domains=domains,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        pretrained=False
    )

    # Check that heads exist for each domain
    for domain in domains:
        assert hasattr(model.heads, domain), f"Model should have head for {domain}"
        head = getattr(model.heads, domain)
        assert isinstance(head, nn.Linear), f"Head for {domain} should be Linear layer"
        assert head.out_features == num_classes, \
            f"Head for {domain} should output {num_classes} classes"

    # Check that heads are different objects
    head1 = getattr(model.heads, domains[0])
    head2 = getattr(model.heads, domains[1])
    assert head1 is not head2, "Domain heads should be separate modules"

    # Check that they have different parameters
    with torch.no_grad():
        head1.weight.fill_(1.0)
        head2.weight.fill_(2.0)

    assert not torch.allclose(head1.weight, head2.weight), \
        "Domain heads should have independent parameters"

    print(f"✓ All {len(domains)} domains have separate heads")


def test_lora_blocks_shared():
    """Test that LoRA blocks exist and can be accessed."""
    num_classes = 10
    domains = ['domain1', 'domain2']
    lora_rank = 4
    lora_alpha = 4.0

    model = ResNet18_EAPH(
        num_classes=num_classes,
        domains=domains,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        pretrained=False
    )

    # Check that lora_blocks exist
    assert hasattr(model, 'lora_blocks'), "Model should have lora_blocks"

    # Count LoRA layers
    lora_count = 0
    for name, module in model.lora_blocks.named_modules():
        if isinstance(module, LoRAConv2d):
            lora_count += 1

    assert lora_count > 0, "Should have at least one LoRA layer"

    print(f"✓ Model has {lora_count} LoRA layers")


def test_model_forward_with_domain():
    """Test that model forward pass works with domain specification."""
    num_classes = 10
    domains = ['domain1', 'domain2']
    lora_rank = 4
    lora_alpha = 4.0

    model = ResNet18_EAPH(
        num_classes=num_classes,
        domains=domains,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        pretrained=False
    )

    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    # Forward pass with each domain
    for domain in domains:
        with torch.no_grad():
            output = model(x, domain=domain)

        expected_shape = (batch_size, num_classes)
        assert output.shape == expected_shape, \
            f"Expected output shape {expected_shape} for {domain}, got {output.shape}"

    print(f"✓ Forward pass works for all domains")


def test_resnet50_instantiation():
    """Test that ResNet50_EAPH can be instantiated with various configs."""
    num_classes = 10
    domains = ['domain1', 'domain2']
    lora_rank = 8
    lora_alpha = 32.0

    model = ResNet50_EAPH(
        num_classes=num_classes,
        domains=domains,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        pretrained=False
    )

    assert model is not None, "Model should be instantiated"
    assert model.num_classes == num_classes
    assert model.domains == domains
    assert model.lora_rank == lora_rank
    assert model.lora_alpha == lora_alpha

    print(f"✓ ResNet50_EAPH instantiated successfully")


def test_resnet50_feature_dim():
    """Test that ResNet50 outputs 2048-d features."""
    num_classes = 10
    domains = ['domain1']
    
    model = ResNet50_EAPH(
        num_classes=num_classes,
        domains=domains,
        pretrained=False
    )

    # Test feature_dim property
    assert hasattr(model, 'feature_dim'), "Model should have feature_dim property"
    assert model.feature_dim == 2048, f"ResNet50 should have 2048-d features, got {model.feature_dim}"

    # Test forward_features output
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        features = model.forward_features(x)

    expected_shape = (batch_size, 2048)
    assert features.shape == expected_shape, \
        f"Expected feature shape {expected_shape}, got {features.shape}"

    print(f"✓ ResNet50 produces 2048-d features correctly")


def test_resnet50_lora_injection():
    """Test that ResNet50 has LoRA modules attached to layer4 Bottleneck blocks."""
    num_classes = 10
    domains = ['domain1', 'domain2']
    
    model = ResNet50_EAPH(
        num_classes=num_classes,
        domains=domains,
        pretrained=False
    )

    # ResNet50 layer4 has 3 Bottleneck blocks
    # Each should have 1 LoRA module (attached to conv3)
    assert hasattr(model, 'lora_blocks'), "Model should have lora_blocks"
    assert len(model.lora_blocks) == 3, f"ResNet50 should have 3 LoRA modules (one per Bottleneck), got {len(model.lora_blocks)}"

    # Check that each is a LoRAConv2d
    from models.resnet50_lora import LoRAConv2d as LoRAConv2d50
    for i, lora_module in enumerate(model.lora_blocks):
        assert isinstance(lora_module, LoRAConv2d50), \
            f"LoRA block {i} should be LoRAConv2d, got {type(lora_module)}"

    print(f"✓ ResNet50 has 3 LoRA modules attached to layer4")


def test_resnet50_parameters_theta():
    """Test that ResNet50 parameters_theta excludes LoRA and heads."""
    num_classes = 10
    domains = ['domain1', 'domain2']
    
    model = ResNet50_EAPH(
        num_classes=num_classes,
        domains=domains,
        pretrained=False
    )

    theta_params = list(model.parameters_theta())
    theta_param_ids = {id(p) for p in theta_params}

    # Check that no LoRA or head parameters are included
    for name, param in model.named_parameters():
        if 'lora_blocks' in name or 'heads' in name:
            assert id(param) not in theta_param_ids, \
                f"Parameter {name} should not be in theta parameters"
        else:
            assert id(param) in theta_param_ids, \
                f"Parameter {name} should be in theta parameters"

    assert len(theta_params) > 0, "Theta parameters should not be empty"

    print(f"✓ ResNet50 theta has {len(theta_params)} parameters (excluding LoRA and heads)")


def test_resnet50_parameters_phi():
    """Test that ResNet50 parameters_phi includes LoRA and domain head."""
    num_classes = 10
    domains = ['domain1', 'domain2']
    
    model = ResNet50_EAPH(
        num_classes=num_classes,
        domains=domains,
        pretrained=False
    )

    for domain in domains:
        phi_params = list(model.parameters_phi(domain))
        phi_param_ids = {id(p) for p in phi_params}

        lora_found = False
        head_found = False

        for name, param in model.named_parameters():
            if 'lora_blocks' in name:
                assert id(param) in phi_param_ids, \
                    f"LoRA parameter {name} should be in phi parameters for {domain}"
                lora_found = True
            elif f'heads.{domain}' in name:
                assert id(param) in phi_param_ids, \
                    f"Head parameter {name} should be in phi parameters for {domain}"
                head_found = True

        assert lora_found, f"LoRA parameters should be in phi for {domain}"
        assert head_found, f"Domain head should be in phi for {domain}"

        print(f"✓ ResNet50 phi for {domain} has {len(phi_params)} parameters")


def test_resnet50_forward():
    """Test that ResNet50 forward pass works with domain specification."""
    num_classes = 10
    domains = ['domain1', 'domain2']
    
    model = ResNet50_EAPH(
        num_classes=num_classes,
        domains=domains,
        pretrained=False
    )

    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    for domain in domains:
        with torch.no_grad():
            output = model(x, domain=domain)

        expected_shape = (batch_size, num_classes)
        assert output.shape == expected_shape, \
            f"Expected output shape {expected_shape} for {domain}, got {output.shape}"

    print(f"✓ ResNet50 forward pass works for all domains")


def test_resnet50_gradient_flow():
    """Test that gradients flow through LoRA modules in ResNet50."""
    num_classes = 10
    domains = ['domain1']
    
    model = ResNet50_EAPH(
        num_classes=num_classes,
        domains=domains,
        pretrained=False
    )

    # Initialize lora_up weights to non-zero values to test gradient flow
    for lora_module in model.lora_blocks:
        lora_module.lora_up.weight.data.normal_(0, 0.01)

    # Create dummy input and target
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    target = torch.randint(0, num_classes, (batch_size,))

    # Forward pass
    output = model(x, domain=domains[0])
    loss = torch.nn.functional.cross_entropy(output, target)

    # Backward pass
    loss.backward()

    # Check that LoRA parameters have gradients
    lora_params_with_grad = 0
    for name, param in model.named_parameters():
        if 'lora_blocks' in name:
            assert param.grad is not None, \
                f"LoRA parameter {name} should have gradient"
            # Check if gradient has non-zero values
            if torch.abs(param.grad).sum() > 1e-10:
                lora_params_with_grad += 1

    # At least some LoRA parameters should have non-zero gradients
    assert lora_params_with_grad > 0, "At least some LoRA parameters should have non-zero gradients"

    print(f"✓ Gradients flow through ResNet50 LoRA modules ({lora_params_with_grad} params with non-zero grads)")


def test_resnet50_domain_heads():
    """Test that ResNet50 has correct domain-specific heads."""
    num_classes = 10
    domains = ['domain1', 'domain2', 'domain3']
    
    model = ResNet50_EAPH(
        num_classes=num_classes,
        domains=domains,
        pretrained=False
    )

    # Check heads exist
    for domain in domains:
        assert hasattr(model.heads, domain), f"Model should have head for {domain}"
        head = getattr(model.heads, domain)
        assert isinstance(head, nn.Linear), f"Head for {domain} should be Linear"
        assert head.in_features == 2048, f"Head should accept 2048-d features"
        assert head.out_features == num_classes, f"Head should output {num_classes} classes"

    print(f"✓ ResNet50 has correct domain-specific heads (2048→{num_classes})")


if __name__ == '__main__':
    # ResNet18 tests
    test_lora_conv2d_residual_path()
    test_parameters_theta_excludes_lora()
    test_parameters_phi_includes_lora()
    test_forward_features()
    test_domain_heads_separate()
    test_lora_blocks_shared()
    test_model_forward_with_domain()
    
    # ResNet50 tests
    print("\n--- ResNet50 Tests ---")
    test_resnet50_instantiation()
    test_resnet50_feature_dim()
    test_resnet50_lora_injection()
    test_resnet50_parameters_theta()
    test_resnet50_parameters_phi()
    test_resnet50_forward()
    test_resnet50_gradient_flow()
    test_resnet50_domain_heads()
    
    print("\nAll model tests passed!")
