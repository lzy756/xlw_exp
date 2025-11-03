"""Metrics calculation utilities for federated learning."""

from typing import Dict, List, Tuple
import numpy as np


def running_accuracy(correct: int, total: int) -> float:
    """Calculate running accuracy.

    Args:
        correct: Number of correct predictions
        total: Total number of predictions

    Returns:
        Accuracy as percentage (0-100)
    """
    if total == 0:
        return 0.0
    return 100.0 * correct / total


def per_domain_metrics(domain_results: Dict[str, float]) -> Tuple[float, float, float]:
    """Calculate aggregate metrics across domains.

    Args:
        domain_results: Dictionary mapping domain name to accuracy

    Returns:
        Tuple of (average_accuracy, worst_accuracy, variance)
    """
    if not domain_results:
        return 0.0, 0.0, 0.0

    accuracies = list(domain_results.values())

    avg_acc = np.mean(accuracies)
    worst_acc = np.min(accuracies)
    variance = np.var(accuracies)

    return float(avg_acc), float(worst_acc), float(variance)