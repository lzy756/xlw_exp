"""Aggregation functions for federated learning."""

from typing import Dict, List, Tuple
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
    domain: str | None = None
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


def _normalize_scores(
    scores: np.ndarray,
    mode: str = "zscore"
) -> np.ndarray:
    """Normalize selector scores for fair-weighted aggregation.

    Args:
        scores: Array of raw selector scores (shape: [num_domains])
        mode: Normalization mode - "zscore" or "minmax"

    Returns:
        Normalized scores (same shape as input)
    
    Raises:
        ValueError: If mode is not "zscore" or "minmax"
    """
    if mode == "zscore":
        mean = np.mean(scores)
        std = np.std(scores)
        if std < 1e-8:  # Handle zero variance
            return np.zeros_like(scores)
        return (scores - mean) / std
    
    elif mode == "minmax":
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val < 1e-8:  # Handle zero range
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}. Expected 'zscore' or 'minmax'.")


def aggregate_theta_weighted(
    theta_e_list: List[Dict[str, torch.Tensor]],
    domains: List[str],
    n_map: Dict[str, int],
    score_map: Dict[str, float],
    agg_domain: str,
    cfg: Dict
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """Fair-weighted aggregation with domain fairness factors.

    Combines sample counts with fairness-aware factors derived from selector scores.
    Gives higher weight to domains with lower scores (struggling domains).

    Args:
        theta_e_list: List of theta state dicts from each domain
        domains: List of domain names (must match theta_e_list order)
        n_map: Dict mapping domain -> sample count used in training
        score_map: Dict mapping domain -> raw selector score (pre-softmax)
        agg_domain: Selected aggregator domain (receives lambda boost)
        cfg: Configuration dict with fair_weighting section

    Returns:
        Tuple of:
            - theta_avg: Aggregated global backbone parameters
            - alpha: Final aggregation weights (length=len(domains))
            - q: Fairness factors (length=len(domains))

    Raises:
        ValueError: If domains are missing from maps or if sample counts are invalid
    """
    # Extract configuration
    fw_cfg = cfg.get('fair_weighting', {})
    beta = fw_cfg.get('beta', 0.5)
    lambda_boost = fw_cfg.get('lambda_boost', 0.2)
    norm_mode = fw_cfg.get('norm', 'zscore')

    # Validate inputs
    for domain in domains:
        if domain not in n_map:
            raise ValueError(f"Domain '{domain}' missing from n_map")
        if domain not in score_map:
            raise ValueError(f"Domain '{domain}' missing from score_map")
        if n_map[domain] < 0:
            raise ValueError(f"Sample count for domain '{domain}' must be non-negative")

    # Build aligned vectors
    n = np.array([n_map[d] for d in domains], dtype=np.float64)
    S = np.array([score_map[d] for d in domains], dtype=np.float64)

    # Validate scores are finite
    if not np.all(np.isfinite(S)):
        raise ValueError("Score values must be finite")

    # 1. Normalize scores
    S_norm = _normalize_scores(S, mode=norm_mode)

    # 2. Add lambda boost to selected aggregator
    if agg_domain in domains:
        agg_idx = domains.index(agg_domain)
        S_norm[agg_idx] += lambda_boost

    # 3. Compute fairness factors via softmax
    # Higher scores get lower weights (invert by negation)
    exp_vals = np.exp(-beta * S_norm)
    q = exp_vals / np.sum(exp_vals)  # Fairness factors

    # 4. Compute final weights: alpha = normalize(n ⊙ q)
    alpha_unnorm = n * q
    alpha_sum = np.sum(alpha_unnorm)
    if alpha_sum < 1e-8:  # Handle all-zero case
        alpha = np.ones(len(domains)) / len(domains)
    else:
        alpha = alpha_unnorm / alpha_sum

    # 5. Weighted averaging of theta states
    weights_list = alpha.tolist()
    theta_avg = aggregate_theta(theta_e_list, weights_list)

    return theta_avg, alpha, q