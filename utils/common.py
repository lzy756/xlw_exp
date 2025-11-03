"""Common utilities for FL-DomainNet experiment."""

import random
import logging
import subprocess
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Build a logger instance.

    Args:
        name: Logger name
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_git_commit_hash() -> str:
    """Get current git commit hash for tracking.

    Returns:
        Git commit hash string or 'unknown' if not in git repo
    """
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return commit_hash[:8]  # Use short hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


class EMA:
    """Exponential Moving Average for tracking statistics.

    Attributes:
        beta: Smoothing factor (0 < beta < 1)
        v: Current EMA value
        initialized: Whether EMA has been initialized with first value
    """

    def __init__(self, beta: float = 0.9):
        """Initialize EMA.

        Args:
            beta: Smoothing factor, higher = more smoothing
        """
        if not 0 < beta < 1:
            raise ValueError(f"Beta must be in (0, 1), got {beta}")

        self.beta = beta
        self.v: Optional[float] = None
        self.initialized = False

    def update(self, x: float) -> float:
        """Update EMA with new value.

        Args:
            x: New value to incorporate

        Returns:
            Updated EMA value
        """
        if not self.initialized:
            self.v = x
            self.initialized = True
        else:
            self.v = self.beta * self.v + (1 - self.beta) * x

        return self.v

    def get(self) -> Optional[float]:
        """Get current EMA value.

        Returns:
            Current EMA value or None if not initialized
        """
        return self.v

    def reset(self) -> None:
        """Reset EMA to uninitialized state."""
        self.v = None
        self.initialized = False