"""Experiment logging and management utilities."""

import os
import sys
import json
import socket
import subprocess
import platform
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path
import yaml
import torch


class ExperimentLogger:
    """Manages experiment directories, configuration saving, and metadata recording.
    
    This class handles:
    - Creating timestamped experiment directories
    - Saving original and effective configurations
    - Recording experiment metadata (git info, environment, timing)
    - Creating symlinks to latest experiment
    
    Attributes:
        config: Experiment configuration dictionary
        args: Command-line arguments
        exp_dir: Path to experiment directory
        start_time: Experiment start timestamp
        info_file: Path to experiment_info.json
    """
    
    def __init__(self, config: Dict, args: Any):
        """Initialize experiment logger.
        
        Args:
            config: Configuration dictionary (should contain 'logging' section)
            args: Parsed command-line arguments
        """
        self.config = config
        self.args = args
        self.exp_dir: Optional[str] = None
        self.start_time = datetime.now()
        self.info_file: Optional[str] = None
        self._command_line = " ".join(sys.argv)
        self._working_dir = os.getcwd()
    
    def create_experiment_dir(self) -> str:
        """Create timestamped experiment directory.
        
        Returns:
            Path to created experiment directory
            
        Raises:
            OSError: If directory creation fails
        """
        logging_config = self.config.get('logging', {})
        
        # Get base output directory
        if hasattr(self.args, 'output_dir') and self.args.output_dir:
            output_base = self.args.output_dir
        else:
            output_base = logging_config.get('output_dir', './outputs')
        
        # Check if timestamp mode is enabled
        use_timestamp = logging_config.get('use_timestamp_dir', True)
        
        # Override with command-line flag if provided
        if hasattr(self.args, 'no_timestamp') and self.args.no_timestamp:
            use_timestamp = False
        
        if use_timestamp:
            # Generate timestamp directory name
            timestamp_format = logging_config.get('timestamp_format', '%Y-%m-%d-%H-%M')
            timestamp_str = self.start_time.strftime(timestamp_format)
            
            # Add optional experiment tag
            exp_tag = None
            if hasattr(self.args, 'exp_tag') and self.args.exp_tag:
                exp_tag = self.args.exp_tag
            elif logging_config.get('exp_name'):
                exp_tag = logging_config['exp_name']
            
            if exp_tag:
                dir_name = f"{timestamp_str}-{exp_tag}"
            else:
                dir_name = timestamp_str
        else:
            # Use fixed experiment name
            exp_name = logging_config.get('exp_name', 'exp1')
            if hasattr(self.args, 'exp_tag') and self.args.exp_tag:
                dir_name = self.args.exp_tag
            else:
                dir_name = exp_name
        
        # Create full path
        self.exp_dir = os.path.join(output_base, dir_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Create checkpoints subdirectory
        os.makedirs(os.path.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        
        # Create symlink to latest experiment
        if use_timestamp:
            self._create_latest_symlink(output_base)
        
        return self.exp_dir
    
    def _create_latest_symlink(self, output_base: str) -> None:
        """Create 'latest' symlink pointing to current experiment.
        
        Args:
            output_base: Base output directory
        """
        latest_link = os.path.join(output_base, 'latest')
        
        # Remove existing symlink if it exists
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            # If it's a regular file/directory, don't remove it
            return
        
        # Create new symlink (relative path)
        try:
            rel_path = os.path.basename(self.exp_dir)
            os.symlink(rel_path, latest_link)
        except OSError as e:
            # Symlink creation may fail on some systems, just log warning
            print(f"Warning: Could not create 'latest' symlink: {e}")
    
    def save_configs(self, original_config: Dict, effective_config: Dict) -> None:
        """Save both original and effective configurations.
        
        Args:
            original_config: Original configuration before overrides
            effective_config: Final configuration after applying overrides
        """
        if not self.exp_dir:
            raise RuntimeError("Experiment directory not created. Call create_experiment_dir() first.")
        
        logging_config = self.config.get('logging', {})
        if not logging_config.get('save_config', True):
            return
        
        # Save original configuration
        original_path = os.path.join(self.exp_dir, 'config.yaml')
        with open(original_path, 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False, sort_keys=False)
        
        # Save effective configuration
        effective_path = os.path.join(self.exp_dir, 'config_effective.yaml')
        with open(effective_path, 'w') as f:
            yaml.dump(effective_config, f, default_flow_style=False, sort_keys=False)
    
    def save_experiment_info(self, status: str = 'running', **kwargs) -> None:
        """Save or update experiment metadata.
        
        Args:
            status: Experiment status ('running', 'completed', 'failed', 'interrupted')
            **kwargs: Additional metadata to include (e.g., error message, exit_code)
        """
        if not self.exp_dir:
            raise RuntimeError("Experiment directory not created. Call create_experiment_dir() first.")
        
        logging_config = self.config.get('logging', {})
        if not logging_config.get('save_experiment_info', True):
            return
        
        self.info_file = os.path.join(self.exp_dir, 'experiment_info.json')
        
        # Load existing info if available
        if os.path.exists(self.info_file):
            with open(self.info_file, 'r') as f:
                info = json.load(f)
        else:
            # Initialize new info
            info = {
                'start_time': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'command': self._command_line,
                'working_directory': self._working_dir,
                'git_info': self.get_git_info(),
                'environment': self.get_environment_info()
            }
        
        # Update status
        info['status'] = status
        
        # Add end time and duration if experiment is finished
        if status in ['completed', 'failed', 'interrupted']:
            end_time = datetime.now()
            info['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')
            info['duration_seconds'] = int((end_time - self.start_time).total_seconds())
        
        # Add any additional kwargs
        info.update(kwargs)
        
        # Save info
        with open(self.info_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def get_git_info(self) -> Dict[str, Any]:
        """Get Git repository information.
        
        Returns:
            Dictionary with git metadata (commit, branch, dirty status, remote)
        """
        git_info = {
            'commit_hash': 'unknown',
            'branch': 'unknown',
            'is_dirty': False,
            'remote_url': 'unknown'
        }
        
        try:
            # Get commit hash
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                cwd=self._working_dir
            ).decode('ascii').strip()
            git_info['commit_hash'] = commit[:8]  # Short hash
            
            # Get branch name
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL,
                cwd=self._working_dir
            ).decode('ascii').strip()
            git_info['branch'] = branch
            
            # Check if working directory is dirty
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL,
                cwd=self._working_dir
            ).decode('ascii').strip()
            git_info['is_dirty'] = len(status) > 0
            
            # Get remote URL
            remote = subprocess.check_output(
                ['git', 'remote', 'get-url', 'origin'],
                stderr=subprocess.DEVNULL,
                cwd=self._working_dir
            ).decode('ascii').strip()
            git_info['remote_url'] = remote
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass  # Not a git repository or git not installed
        
        return git_info
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get environment information.
        
        Returns:
            Dictionary with environment metadata (Python, PyTorch, CUDA versions, etc.)
        """
        env_info = {
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'processor': platform.processor()
        }
        
        # Add CUDA version if available
        if torch.cuda.is_available():
            env_info['cuda_available'] = True
            env_info['cuda_version'] = torch.version.cuda
            env_info['cudnn_version'] = str(torch.backends.cudnn.version())
            env_info['gpu_count'] = torch.cuda.device_count()
            env_info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                     for i in range(torch.cuda.device_count())]
        else:
            env_info['cuda_available'] = False
        
        return env_info
    
    def get_exp_dir(self) -> Optional[str]:
        """Get the experiment directory path.
        
        Returns:
            Path to experiment directory or None if not created yet
        """
        return self.exp_dir
