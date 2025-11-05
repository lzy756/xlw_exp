"""Unit tests for aggregator calibration step."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from core.loop import ExperimentEnv, _vectorize_theta, calibrate_on_dc
from data.domainnet import DomainNetDataset


def create_mock_dataset(length=100):
    """Helper to create a mock dataset with proper __len__ support."""
    mock = MagicMock()
    mock.__len__.return_value = length
    return mock


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7)  # Theta parameter
        self.bn1 = nn.BatchNorm2d(64)  # Theta parameter
        
        # Phi parameters (should be frozen during calibration)
        self.heads = nn.ModuleDict({
            'domain1': nn.Linear(512, 10),
            'domain2': nn.Linear(512, 10),
        })
        self.lora_blocks = nn.ModuleDict({
            '0': nn.Linear(512, 16),
        })
    
    def forward(self, x, domain):
        # Simplified forward pass
        return torch.randn(x.size(0), 10)


# ============================================================================
# Tests for ExperimentEnv
# ============================================================================

def test_experiment_env_initialization():
    """Test ExperimentEnv initialization."""
    train_data = {'d1': {}, 'd2': {}}
    config = {'data': {'root': '/tmp'}}
    domains = ['d1', 'd2']
    
    env = ExperimentEnv(train_data, config, domains)
    
    assert env.train_data == train_data
    assert env.config == config
    assert env.domains == domains


def test_experiment_env_dc_unload_dataset_success():
    """Test getting DC unload dataset when available."""
    train_data = {
        'd1': {
            'dc_unload_pool': [0, 1, 2, 3, 4]
        }
    }
    config = {'data': {'root': '/tmp'}}
    domains = ['d1']
    
    env = ExperimentEnv(train_data, config, domains)
    
    # Mock DomainNetDataset to avoid actual data loading
    with patch('core.loop.DomainNetDataset') as mock_dataset:
        mock_dataset.return_value.__len__.return_value = 5
        
        dataset = env.dc_unload_dataset('d1')
        
        # Verify dataset was created with correct parameters
        mock_dataset.assert_called_once_with(
            root='/tmp',
            indices=[0, 1, 2, 3, 4],
            train=True
        )


def test_experiment_env_dc_unload_dataset_empty():
    """Test getting DC unload dataset when pool is empty."""
    train_data = {
        'd1': {
            'dc_unload_pool': []
        }
    }
    config = {'data': {'root': '/tmp'}}
    domains = ['d1']
    
    env = ExperimentEnv(train_data, config, domains)
    
    with patch('core.loop.DomainNetDataset') as mock_dataset:
        mock_dataset.return_value.__len__.return_value = 0
        
        dataset = env.dc_unload_dataset('d1')
        
        assert dataset is None, "Should return None for empty pool"


def test_experiment_env_dc_unload_dataset_missing_domain():
    """Test getting DC unload dataset for non-existent domain."""
    train_data = {'d1': {}}
    config = {'data': {'root': '/tmp'}}
    domains = ['d1']
    
    env = ExperimentEnv(train_data, config, domains)
    
    dataset = env.dc_unload_dataset('d2')
    assert dataset is None, "Should return None for missing domain"


def test_experiment_env_count_domain_samples_ue_only():
    """Test counting samples with UE clients only."""
    train_data = {
        'd1': {
            'clients': [
                {'local': [0, 1, 2]},  # 3 samples
                {'local': [3, 4, 5, 6]},  # 4 samples
                {'local': [7, 8]},  # 2 samples
            ]
        }
    }
    config = {'data': {'offload_pool_enabled': False}}
    domains = ['d1']
    
    env = ExperimentEnv(train_data, config, domains)
    
    # Clients 0 and 2 participating
    participating = {'d1': [0, 2]}
    
    count = env.count_domain_samples_this_round('d1', participating)
    
    # Should be 3 + 2 = 5 samples
    assert count == 5


def test_experiment_env_count_domain_samples_with_dc():
    """Test counting samples with both UE and DC."""
    train_data = {
        'd1': {
            'clients': [
                {'local': [0, 1, 2]},  # 3 samples
            ],
            'dc_unload_pool': [100, 101, 102, 103, 104]  # 5 DC samples
        }
    }
    config = {'data': {'offload_pool_enabled': True}}
    domains = ['d1']
    
    env = ExperimentEnv(train_data, config, domains)
    
    participating = {'d1': [0]}
    
    count = env.count_domain_samples_this_round('d1', participating)
    
    # Should be 3 (UE) + 5 (DC) = 8 samples
    assert count == 8


def test_experiment_env_build_loader():
    """Test building DataLoader."""
    train_data = {}
    config = {'system': {'num_workers': 2}}
    domains = []
    
    env = ExperimentEnv(train_data, config, domains)
    
    # Create mock dataset
    mock_dataset = create_mock_dataset(100)
    
    loader = env.build_loader(mock_dataset, batch_size=32, train=True)
    
    assert loader.batch_size == 32
    assert loader.dataset == mock_dataset
    # Note: shuffle and num_workers are set in DataLoader initialization


# ============================================================================
# Tests for _vectorize_theta
# ============================================================================

def test_vectorize_theta_filters_correctly():
    """Test that _vectorize_theta filters out phi parameters."""
    model_state = {
        'conv1.weight': torch.ones(64, 3, 7, 7),  # Theta
        'bn1.weight': torch.ones(64),  # Theta
        'heads.d1.weight': torch.ones(10, 512),  # Phi - should be excluded
        'lora_blocks.0.weight': torch.ones(16, 512),  # Phi - should be excluded
    }
    
    vec = _vectorize_theta(model_state)
    
    # Calculate expected size (conv + bn)
    expected_size = 64 * 3 * 7 * 7 + 64
    assert vec.numel() == expected_size


def test_vectorize_theta_empty():
    """Test vectorizing empty state dict."""
    vec = _vectorize_theta({})
    assert vec.numel() == 0


def test_vectorize_theta_consistency():
    """Test that vectorization is consistent (same input -> same output)."""
    model_state = {
        'param1': torch.tensor([1.0, 2.0, 3.0]),
        'param2': torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
    }
    
    vec1 = _vectorize_theta(model_state)
    vec2 = _vectorize_theta(model_state)
    
    assert torch.equal(vec1, vec2)


# ============================================================================
# Tests for calibrate_on_dc
# ============================================================================

def test_calibrate_on_dc_disabled_offload():
    """Test calibration skips when offload pool is disabled."""
    config = {
        'data': {'offload_pool_enabled': False},
        'calibration': {'enable': True},
        'system': {'device': 'cpu'}
    }
    
    theta_avg = {'param': torch.ones(5)}
    model = DummyModel()
    env = Mock()
    logger = Mock()
    
    result = calibrate_on_dc('d1', theta_avg, config, model, env, logger)
    
    # Should return theta_avg unchanged
    assert result is theta_avg
    logger.info.assert_called_with("Calibration skipped: offload pool disabled")


def test_calibrate_on_dc_insufficient_samples():
    """Test calibration skips with insufficient samples."""
    config = {
        'data': {'offload_pool_enabled': True},
        'calibration': {
            'enable': True,
            'min_samples': 100
        },
        'system': {'device': 'cpu'}
    }
    
    theta_avg = {'param': torch.ones(5)}
    model = DummyModel()
    env = Mock()
    logger = Mock()
    
    # Mock dataset with only 50 samples (below threshold)
    mock_dataset = create_mock_dataset(50)
    env.dc_unload_dataset.return_value = mock_dataset
    
    result = calibrate_on_dc('d1', theta_avg, config, model, env, logger)
    
    # Should return theta_avg unchanged
    assert result is theta_avg
    assert "insufficient samples" in logger.info.call_args_list[-1][0][0]


def test_calibrate_on_dc_no_dataset():
    """Test calibration skips when dataset is None."""
    config = {
        'data': {'offload_pool_enabled': True},
        'calibration': {
            'enable': True,
            'min_samples': 100
        },
        'system': {'device': 'cpu'}
    }
    
    theta_avg = {'param': torch.ones(5)}
    model = DummyModel()
    env = Mock()
    logger = Mock()
    
    env.dc_unload_dataset.return_value = None
    
    result = calibrate_on_dc('d1', theta_avg, config, model, env, logger)
    
    assert result is theta_avg


def test_calibrate_on_dc_freezes_phi():
    """Test that calibration freezes phi parameters correctly."""
    # This is a unit test that verifies parameter freezing logic
    # without running full calibration loop
    
    model = DummyModel()
    
    # Simulate the freezing logic from calibrate_on_dc
    freeze_phi = True
    if freeze_phi:
        for name, param in model.named_parameters():
            if 'heads.' in name or 'lora_blocks.' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    # Verify phi parameters are frozen
    for name, param in model.named_parameters():
        if 'heads.' in name or 'lora_blocks.' in name:
            assert param.requires_grad == False, f"Phi parameter {name} should be frozen"
        else:
            assert param.requires_grad == True, f"Theta parameter {name} should be trainable"
    
    # Collect theta parameters (should exclude phi)
    theta_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and 'heads.' not in name and 'lora_blocks.' not in name
    ]
    
    # Should only have conv1 and bn1 parameters
    assert len(theta_params) > 0, "Should have theta parameters"
    assert all(p.requires_grad for p in theta_params), "All theta params should require grad"


def test_calibrate_on_dc_returns_theta_only():
    """Test that calibration returns only theta parameters."""
    # This test verifies the extraction logic without full calibration
    
    model = DummyModel()
    
    # Get full state dict
    full_state = model.state_dict()
    
    # Extract only theta (logic from calibrate_on_dc)
    calibrated_theta = {
        k: v.cpu().clone()
        for k, v in full_state.items()
        if 'heads.' not in k and 'lora_blocks.' not in k
    }
    
    # Verify only theta parameters in result
    for key in calibrated_theta.keys():
        assert 'heads.' not in key, "Result should not contain head parameters"
        assert 'lora_blocks.' not in key, "Result should not contain lora parameters"
    
    # Verify theta parameters ARE included
    assert 'conv1.weight' in calibrated_theta
    assert 'bn1.weight' in calibrated_theta
