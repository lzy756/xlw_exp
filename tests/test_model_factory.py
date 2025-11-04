"""Unit tests for model factory and configuration."""

import pytest
import torch
from run_experiment import create_model, load_config


def test_create_model_resnet18():
    """Test that create_model creates ResNet18 when backbone='resnet18'."""
    config = {
        'model': {
            'backbone': 'resnet18',
            'lora': {
                'rank': 16,
                'alpha': 16.0
            },
            'pretrained': False
        },
        'data': {
            'num_classes': 10,
            'domains': ['domain1', 'domain2']
        }
    }

    model = create_model(config)

    # Should be ResNet18_EAPH
    from models.resnet18_lora import ResNet18_EAPH
    assert isinstance(model, ResNet18_EAPH), \
        f"Expected ResNet18_EAPH, got {type(model)}"

    # Check feature dimension
    assert model.feature_dim == 512, \
        f"ResNet18 should have 512-d features, got {model.feature_dim}"

    print(f"✓ create_model creates ResNet18 correctly")


def test_create_model_resnet50():
    """Test that create_model creates ResNet50 when backbone='resnet50'."""
    config = {
        'model': {
            'backbone': 'resnet50',
            'lora': {
                'rank': 16,
                'alpha': 32.0
            },
            'pretrained': False
        },
        'data': {
            'num_classes': 10,
            'domains': ['domain1', 'domain2']
        }
    }

    model = create_model(config)

    # Should be ResNet50_EAPH
    from models.resnet50_lora import ResNet50_EAPH
    assert isinstance(model, ResNet50_EAPH), \
        f"Expected ResNet50_EAPH, got {type(model)}"

    # Check feature dimension
    assert model.feature_dim == 2048, \
        f"ResNet50 should have 2048-d features, got {model.feature_dim}"

    print(f"✓ create_model creates ResNet50 correctly")


def test_create_model_default_backbone():
    """Test that create_model defaults to resnet18 if backbone not specified."""
    config = {
        'model': {
            # No 'backbone' field
            'lora': {
                'rank': 16,
                'alpha': 16.0
            },
            'pretrained': False
        },
        'data': {
            'num_classes': 10,
            'domains': ['domain1', 'domain2']
        }
    }

    model = create_model(config)

    # Should default to ResNet18
    from models.resnet18_lora import ResNet18_EAPH
    assert isinstance(model, ResNet18_EAPH), \
        "Should default to ResNet18 when backbone not specified"

    print(f"✓ create_model defaults to ResNet18 when backbone not specified")


def test_create_model_invalid_backbone():
    """Test that create_model raises ValueError for invalid backbone."""
    config = {
        'model': {
            'backbone': 'resnet101',  # Not supported
            'lora': {
                'rank': 16,
                'alpha': 16.0
            },
            'pretrained': False
        },
        'data': {
            'num_classes': 10,
            'domains': ['domain1', 'domain2']
        }
    }

    with pytest.raises(ValueError) as excinfo:
        model = create_model(config)

    assert 'Unsupported backbone' in str(excinfo.value), \
        "Should raise ValueError with 'Unsupported backbone' message"
    assert 'resnet101' in str(excinfo.value), \
        "Error message should mention the invalid backbone"

    print(f"✓ create_model raises ValueError for invalid backbone")


def test_create_model_passes_config_params():
    """Test that create_model passes all config parameters correctly."""
    config = {
        'model': {
            'backbone': 'resnet50',
            'lora': {
                'rank': 8,
                'alpha': 24.0
            },
            'pretrained': False
        },
        'data': {
            'num_classes': 20,
            'domains': ['d1', 'd2', 'd3']
        }
    }

    model = create_model(config)

    # Check that parameters were passed correctly
    assert model.num_classes == 20
    assert model.domains == ['d1', 'd2', 'd3']
    assert model.lora_rank == 8
    assert model.lora_alpha == 24.0

    print(f"✓ create_model passes config parameters correctly")


def test_resnet50_config_file():
    """Test that ResNet50 config file can be loaded and has correct values."""
    import os
    config_path = 'configs/resnet50.yaml'
    
    if not os.path.exists(config_path):
        pytest.skip(f"Config file {config_path} not found")
    
    config = load_config(config_path)

    # Check key ResNet50-specific values
    assert config['model']['backbone'] == 'resnet50', \
        "ResNet50 config should have backbone='resnet50'"
    assert config['model']['lora']['alpha'] == 32, \
        "ResNet50 config should have lora.alpha=32"
    assert config['training']['batch_size'] == 48, \
        "ResNet50 config should have batch_size=48 for memory management"

    print(f"✓ ResNet50 config file has correct values")


def test_model_factory_integration():
    """Integration test: create both models and verify they work end-to-end."""
    # Test ResNet18
    config_18 = {
        'model': {
            'backbone': 'resnet18',
            'lora': {'rank': 16, 'alpha': 16.0},
            'pretrained': False
        },
        'data': {
            'num_classes': 10,
            'domains': ['d1', 'd2']
        }
    }

    model_18 = create_model(config_18)
    
    # Test ResNet50
    config_50 = {
        'model': {
            'backbone': 'resnet50',
            'lora': {'rank': 16, 'alpha': 32.0},
            'pretrained': False
        },
        'data': {
            'num_classes': 10,
            'domains': ['d1', 'd2']
        }
    }

    model_50 = create_model(config_50)

    # Test forward pass on both
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out_18 = model_18(x, domain='d1')
        out_50 = model_50(x, domain='d1')

    assert out_18.shape == (2, 10)
    assert out_50.shape == (2, 10)

    print(f"✓ Both models work end-to-end via factory")


if __name__ == '__main__':
    test_create_model_resnet18()
    test_create_model_resnet50()
    test_create_model_default_backbone()
    test_create_model_invalid_backbone()
    test_create_model_passes_config_params()
    test_resnet50_config_file()
    test_model_factory_integration()

    print("\nAll model factory tests passed!")
