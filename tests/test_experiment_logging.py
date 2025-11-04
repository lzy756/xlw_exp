"""Tests for experiment logging and metadata."""

import pytest
import json
import os
import tempfile
import shutil
from pathlib import Path
from utils.experiment import ExperimentLogger
import yaml


class MockArgs:
    """Mock arguments for testing."""
    def __init__(self, output_dir=None, no_timestamp=False, exp_tag=None):
        self.output_dir = output_dir
        self.no_timestamp = no_timestamp
        self.exp_tag = exp_tag


def test_experiment_logger_creates_directory():
    """Test that ExperimentLogger creates experiment directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'logging': {
                'output_dir': tmpdir,
                'use_timestamp_dir': False,
                'exp_name': 'test_exp'
            }
        }
        args = MockArgs()

        exp_logger = ExperimentLogger(config, args)
        exp_dir = exp_logger.create_experiment_dir()

        # Check directory exists
        assert os.path.exists(exp_dir)
        assert os.path.isdir(exp_dir)

        # Check checkpoints subdirectory exists
        checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
        assert os.path.exists(checkpoints_dir)
        assert os.path.isdir(checkpoints_dir)


def test_experiment_logger_saves_configs():
    """Test that ExperimentLogger saves configuration files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'logging': {
                'output_dir': tmpdir,
                'use_timestamp_dir': False,
                'exp_name': 'test_exp'
            },
            'training': {
                'rounds': 100
            }
        }
        args = MockArgs()

        exp_logger = ExperimentLogger(config, args)
        exp_dir = exp_logger.create_experiment_dir()

        # Save configs
        original_config = {'training': {'rounds': 50}}
        effective_config = {'training': {'rounds': 100}}

        exp_logger.save_configs(original_config, effective_config)

        # Check files exist
        config_path = os.path.join(exp_dir, 'config.yaml')
        config_eff_path = os.path.join(exp_dir, 'config_effective.yaml')

        assert os.path.exists(config_path)
        assert os.path.exists(config_eff_path)

        # Check content
        with open(config_path, 'r') as f:
            saved_original = yaml.safe_load(f)
        assert saved_original['training']['rounds'] == 50

        with open(config_eff_path, 'r') as f:
            saved_effective = yaml.safe_load(f)
        assert saved_effective['training']['rounds'] == 100


def test_experiment_logger_saves_info():
    """Test that ExperimentLogger saves experiment info with metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'logging': {
                'output_dir': tmpdir,
                'use_timestamp_dir': False,
                'exp_name': 'test_exp'
            }
        }
        args = MockArgs()

        exp_logger = ExperimentLogger(config, args)
        exp_dir = exp_logger.create_experiment_dir()

        # Save initial info
        exp_logger.save_experiment_info(status='running')

        # Check file exists
        info_path = os.path.join(exp_dir, 'experiment_info.json')
        assert os.path.exists(info_path)

        # Check content
        with open(info_path, 'r') as f:
            info = json.load(f)

        assert info['status'] == 'running'
        assert 'start_time' in info
        assert 'command' in info
        assert 'working_directory' in info
        assert 'git_info' in info
        assert 'environment' in info

        # Update info to completed
        exp_logger.save_experiment_info(status='completed', exit_code=0)

        with open(info_path, 'r') as f:
            info = json.load(f)

        assert info['status'] == 'completed'
        assert info['exit_code'] == 0
        assert 'end_time' in info
        assert 'duration_seconds' in info


def test_experiment_logger_git_info():
    """Test that ExperimentLogger captures git information."""
    config = {'logging': {}}
    args = MockArgs()

    exp_logger = ExperimentLogger(config, args)
    git_info = exp_logger.get_git_info()

    # Should return dict with expected keys
    assert 'commit_hash' in git_info
    assert 'branch' in git_info
    assert 'is_dirty' in git_info
    assert 'remote_url' in git_info

    # Values might be 'unknown' if not in git repo, but should exist
    assert isinstance(git_info['commit_hash'], str)
    assert isinstance(git_info['branch'], str)
    assert isinstance(git_info['is_dirty'], bool)
    assert isinstance(git_info['remote_url'], str)


def test_experiment_logger_environment_info():
    """Test that ExperimentLogger captures environment information."""
    config = {'logging': {}}
    args = MockArgs()

    exp_logger = ExperimentLogger(config, args)
    env_info = exp_logger.get_environment_info()

    # Should have basic environment info
    assert 'python_version' in env_info
    assert 'pytorch_version' in env_info
    assert 'hostname' in env_info
    assert 'platform' in env_info
    assert 'cuda_available' in env_info

    assert isinstance(env_info['python_version'], str)
    assert isinstance(env_info['pytorch_version'], str)
    assert isinstance(env_info['cuda_available'], bool)


def test_experiment_logger_with_timestamp():
    """Test that ExperimentLogger creates timestamped directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'logging': {
                'output_dir': tmpdir,
                'use_timestamp_dir': True,
                'timestamp_format': '%Y-%m-%d'
            }
        }
        args = MockArgs()

        exp_logger = ExperimentLogger(config, args)
        exp_dir = exp_logger.create_experiment_dir()

        # Directory name should contain date
        dir_name = os.path.basename(exp_dir)
        assert len(dir_name) >= 10  # At least YYYY-MM-DD format

        # Check latest symlink exists (may not on all systems)
        latest_link = os.path.join(tmpdir, 'latest')
        # Just check it doesn't error, may not exist on all platforms


def test_experiment_logger_with_exp_tag():
    """Test that ExperimentLogger uses experiment tags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'logging': {
                'output_dir': tmpdir,
                'use_timestamp_dir': True,
                'timestamp_format': '%Y-%m-%d'
            }
        }
        args = MockArgs(exp_tag='my-test-run')

        exp_logger = ExperimentLogger(config, args)
        exp_dir = exp_logger.create_experiment_dir()

        # Directory name should contain tag
        dir_name = os.path.basename(exp_dir)
        assert 'my-test-run' in dir_name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
