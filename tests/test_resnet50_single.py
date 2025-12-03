"""Test script to verify ResNet50Single model works correctly."""

import torch
from baseline.models.resnet50_single import ResNet50Single

def test_resnet50_single():
    """Test ResNet50Single model initialization and forward pass."""
    print("Testing ResNet50Single model...")

    # Create model
    model = ResNet50Single(num_classes=126, pretrained=True)
    print(f"✓ Model created successfully")
    print(f"  Feature dimension: {model.feature_dim}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    # Test forward_features
    features = model.forward_features(x)
    print(f"✓ forward_features works")
    print(f"  Input shape: {x.shape}")
    print(f"  Features shape: {features.shape}")
    assert features.shape == (batch_size, 2048), f"Expected (4, 2048), got {features.shape}"

    # Test forward
    logits = model(x)
    print(f"✓ forward works")
    print(f"  Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, 126), f"Expected (4, 126), got {logits.shape}"

    # Test state_dict methods
    state_dict = model.state_dict_global()
    print(f"✓ state_dict_global works")
    print(f"  Number of parameters: {len(state_dict)}")

    model.load_state_dict_global(state_dict)
    print(f"✓ load_state_dict_global works")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Parameter count:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n✅ All tests passed!")

if __name__ == '__main__':
    test_resnet50_single()
