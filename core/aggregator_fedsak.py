"""FedSAK aggregation: Tensor trace norm regularization.

Unlike FedAvg which computes weighted average, FedSAK stacks client parameters
into tensors and applies trace norm regularization to discover low-rank
correlations between clients.
"""

from typing import List, Dict
import torch
from core.fedsak_ops import compute_trace_norm_gradient


def aggregate_fedsak(
    client_states: List[Dict[str, torch.Tensor]],
    lr_w: float = 0.01,
    lam: float = 1.0,
    device: str = 'cuda'
) -> List[Dict[str, torch.Tensor]]:
    """FedSAK aggregation via tensor trace norm regularization.

    Key difference from FedAvg:
    - FedAvg: returns single averaged state dict
    - FedSAK: returns list of regularized state dicts (one per client)

    Algorithm:
    1. Stack client params into tensor W (last dim = client ID)
    2. Compute trace norm gradient: grad = d(||W||_*)/dW
    3. Update: W_new = W - lr_w * lambda * grad
    4. Split back to per-client dicts

    Args:
        client_states: List of state dicts from clients
        lr_w: Server learning rate (eta_w in paper)
        lam: Regularization coefficient (lambda in paper)
        device: Device for computation

    Returns:
        List of updated state dicts (one per client)
    """
    if not client_states:
        raise ValueError("Empty client_states")

    num_clients = len(client_states)
    layer_keys = list(client_states[0].keys())

    # Initialize output
    updated_states = [{} for _ in range(num_clients)]

    # Process each layer independently
    for key in layer_keys:
        # 1. Collect params from all clients
        params_list = [c[key].to(device) for c in client_states]

        # Check if all params have the same shape
        param_shape = params_list[0].shape
        if not all(p.shape == param_shape for p in params_list):
            # Skip layers with mismatched shapes (shouldn't happen in DH)
            for i in range(num_clients):
                updated_states[i][key] = client_states[i][key].cpu().clone()
            continue

        # 2. Stack into tensor (last dim = client ID)
        # Shape: (*param_shape, num_clients)
        W_tensor = torch.stack(params_list, dim=-1)

        # 3. Compute trace norm gradient
        grad_tensor = compute_trace_norm_gradient(W_tensor, device)

        # 4. Regularized update: w_new = w - lr_w * lambda * grad
        W_updated = W_tensor - lr_w * lam * grad_tensor

        # 5. Split back to individual clients
        for i in range(num_clients):
            updated_states[i][key] = W_updated[..., i].cpu().clone()

    return updated_states


def aggregate_fedsak_weighted(
    client_states: List[Dict[str, torch.Tensor]],
    client_weights: List[float],
    lr_w: float = 0.01,
    lam: float = 1.0,
    device: str = 'cuda'
) -> List[Dict[str, torch.Tensor]]:
    """FedSAK aggregation with client weighting (by data size).

    Extended version that considers client data sizes. The trace norm
    regularization is applied, but the initial tensor stacking can be
    weighted to give more importance to clients with more data.

    Note: Standard FedSAK doesn't use weighting, but this is provided
    for experimental comparison.

    Args:
        client_states: List of state dicts
        client_weights: List of weights (e.g., data sizes)
        lr_w: Server learning rate
        lam: Regularization coefficient
        device: Device

    Returns:
        List of updated state dicts
    """
    # Normalize weights
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]

    num_clients = len(client_states)
    layer_keys = list(client_states[0].keys())
    updated_states = [{} for _ in range(num_clients)]

    for key in layer_keys:
        params_list = [c[key].to(device) for c in client_states]
        param_shape = params_list[0].shape

        if not all(p.shape == param_shape for p in params_list):
            for i in range(num_clients):
                updated_states[i][key] = client_states[i][key].cpu().clone()
            continue

        # Weight-adjusted stacking (experimental)
        # Each client's params are scaled by sqrt(weight) before stacking
        # This makes the trace norm regularization weight-aware
        weighted_params = [
            p * (w ** 0.5) for p, w in zip(params_list, normalized_weights)
        ]
        W_tensor = torch.stack(weighted_params, dim=-1)

        grad_tensor = compute_trace_norm_gradient(W_tensor, device)
        W_updated = W_tensor - lr_w * lam * grad_tensor

        # Unscale back
        for i in range(num_clients):
            updated_states[i][key] = (
                W_updated[..., i] / (normalized_weights[i] ** 0.5)
            ).cpu().clone()

    return updated_states
