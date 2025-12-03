"""Aggregation functions for federated learning (v2).

V2 simplified aggregation:
- No LoRA support
- φ only contains domain heads
"""

from typing import Dict, List, Optional
import torch
import numpy as np


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
    # Filter to ensure we only aggregate theta parameters (no heads)
    filtered_theta_list = []
    for theta_state in client_theta_list:
        filtered = {
            k: v for k, v in theta_state.items()
            if 'heads.' not in k
        }
        filtered_theta_list.append(filtered)

    return fedavg(filtered_theta_list, client_weights)


def aggregate_phi_domain(
    client_phi_list: List[Dict[str, torch.Tensor]],
    client_weights: List[float],
    domain: str
) -> Dict[str, torch.Tensor]:
    """Aggregate domain-specific head parameters (φ_e).

    In v2, φ only contains domain heads (no LoRA).

    Args:
        client_phi_list: List of phi state dicts from clients in a domain
        client_weights: Weights for each client
        domain: Domain name for filtering

    Returns:
        Aggregated phi state dictionary for the domain
    """
    # Filter to ensure we only aggregate phi parameters for this domain
    filtered_phi_list = []
    for phi_state in client_phi_list:
        filtered = {
            k: v for k, v in phi_state.items()
            if f'heads.{domain}' in k
        }
        filtered_phi_list.append(filtered)

    return fedavg(filtered_phi_list, client_weights)


def fair_weighted_aggregate(
    domain_theta_updates: Dict[str, List[Dict[str, torch.Tensor]]],
    domain_theta_weights: Dict[str, List[float]],
    domains: List[str],
    fair_factors: Dict[str, float],
    cfg: Optional[Dict] = None
) -> Dict[str, torch.Tensor]:
    """Fair-weighted theta aggregation.
    
    Steps:
    1. Aggregate θ within each domain
    2. Use fair factors to compute final cross-domain weights
    3. Final weight = sample_weight × fair_factor
    4. Cross-domain weighted aggregation
    
    Args:
        domain_theta_updates: Dict mapping domain -> list of theta state dicts
        domain_theta_weights: Dict mapping domain -> list of sample weights
        domains: List of domain names
        fair_factors: Dict mapping domain -> fairness factor (from selector)
        cfg: Optional configuration dict
        
    Returns:
        Aggregated global theta parameters
    """
    # Step 1: Aggregate theta within each domain
    domain_theta = {}
    domain_sample_counts = {}
    
    for domain in domains:
        if domain_theta_updates.get(domain):
            domain_theta[domain] = aggregate_theta(
                domain_theta_updates[domain],
                domain_theta_weights[domain]
            )
            domain_sample_counts[domain] = sum(domain_theta_weights[domain])
    
    if not domain_theta:
        return {}
    
    # Step 2: Compute final weights
    total_samples = sum(domain_sample_counts.values())
    final_weights = {}
    
    for domain in domain_theta:
        base_weight = domain_sample_counts[domain] / total_samples
        fair_factor = fair_factors.get(domain, 1.0 / len(domains))
        final_weights[domain] = base_weight * fair_factor
    
    # Normalize final weights
    total_weight = sum(final_weights.values())
    if total_weight > 0:
        final_weights = {d: w / total_weight for d, w in final_weights.items()}
    else:
        # Fallback to uniform weights
        n = len(final_weights)
        final_weights = {d: 1.0 / n for d in final_weights}
    
    # Step 3: Cross-domain weighted aggregation
    active_domains = list(domain_theta.keys())
    theta_list = [domain_theta[d] for d in active_domains]
    weights = [final_weights[d] for d in active_domains]
    
    return fedavg(theta_list, weights)
