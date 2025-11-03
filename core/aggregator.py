"""Aggregation functions for federated learning."""

from typing import Dict, List
import torch


def fedavg(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float]
) -> Dict[str, torch.Tensor]:
    """Perform weighted FedAvg aggregation.

    Args:
        state_dicts: List of model state dictionaries
        weights: List of weights for each state dict (should sum to 1.0)

    Returns:
        Aggregated state dictionary
    """
    if not state_dicts:
        return {}

    if len(state_dicts) != len(weights):
        raise ValueError(
            f"Number of state dicts ({len(state_dicts)}) must match "
            f"number of weights ({len(weights)})"
        )

    # Normalize weights to ensure they sum to 1
    weight_sum = sum(weights)
    if weight_sum == 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / weight_sum for w in weights]

    # Initialize aggregated state dict
    aggregated = {}

    # Get all parameter keys from first state dict
    for key in state_dicts[0].keys():
        # Initialize with zeros of same shape
        aggregated[key] = torch.zeros_like(state_dicts[0][key])

        # Weighted sum
        for state_dict, weight in zip(state_dicts, weights):
            if key in state_dict:
                aggregated[key] = aggregated[key] + weight * state_dict[key]

    return aggregated


def aggregate_theta(
    client_theta_list: List[Dict[str, torch.Tensor]],
    client_weights: List[float]
) -> Dict[str, torch.Tensor]:
    """Aggregate global backbone parameters (θ).

    Args:
        client_theta_list: List of theta state dicts from clients
        client_weights: Weights for each client (e.g., based on data size)

    Returns:
        Aggregated theta state dictionary
    """
    # Filter to ensure we only aggregate theta parameters
    filtered_theta_list = []
    for theta_state in client_theta_list:
        filtered = {
            k: v for k, v in theta_state.items()
            if 'heads.' not in k and 'lora_blocks.' not in k
        }
        filtered_theta_list.append(filtered)

    return fedavg(filtered_theta_list, client_weights)


def aggregate_phi_domain(
    client_phi_list: List[Dict[str, torch.Tensor]],
    client_weights: List[float],
    domain: str = None
) -> Dict[str, torch.Tensor]:
    """Aggregate domain-specific parameters (φ_e).

    Args:
        client_phi_list: List of phi state dicts from clients in a domain
        client_weights: Weights for each client
        domain: Optional domain name for filtering

    Returns:
        Aggregated phi state dictionary for the domain
    """
    # Filter to ensure we only aggregate phi parameters
    filtered_phi_list = []
    for phi_state in client_phi_list:
        filtered = {}
        for k, v in phi_state.items():
            # Include LoRA blocks (shared across domain but domain-specific weights)
            if 'lora_blocks.' in k:
                filtered[k] = v
            # Include domain-specific head if domain specified
            elif domain and f'heads.{domain}' in k:
                filtered[k] = v
            # Include all heads if no specific domain
            elif not domain and 'heads.' in k:
                filtered[k] = v

        filtered_phi_list.append(filtered)

    return fedavg(filtered_phi_list, client_weights)