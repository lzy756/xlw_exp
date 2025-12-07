"""FedSAK tensor operations: trace norm gradient computation.

Implements tensor unfolding/folding and trace norm gradient for FedSAK aggregation.
Based on Tucker trace norm definition (sum of nuclear norms across all modes).
"""

import torch
from typing import Tuple


def unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """Unfold tensor along mode into matrix.
    
    Args:
        tensor: Input tensor of shape (d1, d2, ..., dp)
        mode: Mode to unfold along (0 to p-1)
    
    Returns:
        Matrix of shape (d_mode, prod(d_i for i != mode))
    """
    return torch.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def fold(matrix: torch.Tensor, mode: int, original_shape: Tuple[int, ...]) -> torch.Tensor:
    """Fold matrix back to tensor.
    
    Args:
        matrix: Input matrix
        mode: Mode that was unfolded
        original_shape: Original tensor shape
    
    Returns:
        Tensor with original_shape
    """
    moved_shape = list(original_shape)
    moved_shape.pop(mode)
    moved_shape.insert(0, original_shape[mode])
    return matrix.reshape(moved_shape).moveaxis(0, mode)


def compute_trace_norm_gradient(W_tensor: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """Compute gradient of tensor trace norm (Tucker definition).
    
    Trace norm: ||W||_* = sum_k ||W_(k)||_*
    where W_(k) is the mode-k unfolding and ||.||_* is nuclear norm.
    
    Gradient (subgradient): sum_k fold_k(U_k @ V_k^T)
    where U_k, V_k come from SVD of W_(k).
    
    Args:
        W_tensor: Parameter tensor of shape (*param_shape, num_clients)
        device: Device for computation
    
    Returns:
        Gradient tensor of same shape as W_tensor
    """
    W_tensor = W_tensor.to(device)
    grad = torch.zeros_like(W_tensor)
    p = W_tensor.dim()  # Tensor order
    
    for k in range(p):
        # 1. Unfold along mode k
        W_k = unfold(W_tensor, k)
        
        # 2. SVD decomposition
        try:
            U, S, Vh = torch.linalg.svd(W_k, full_matrices=False)
            V = Vh.T
        except RuntimeError as e:
            # Fallback: add small regularization if SVD fails
            W_k_reg = W_k + 1e-6 * torch.randn_like(W_k)
            U, S, Vh = torch.linalg.svd(W_k_reg, full_matrices=False)
            V = Vh.T
        
        # 3. Subgradient: U @ V^T (nuclear norm gradient)
        grad_k_matrix = torch.mm(U, V.T)
        
        # 4. Fold back to tensor
        grad_k_tensor = fold(grad_k_matrix, k, W_tensor.shape)
        
        # 5. Accumulate gradient
        grad += grad_k_tensor
    
    return grad


def compute_trace_norm_gradient_selective(
    W_tensor: torch.Tensor,
    device: str = 'cuda',
    skip_last_mode: bool = True
) -> torch.Tensor:
    """Compute trace norm gradient but skip the client dimension.
    
    For efficiency, we often skip regularizing along the client dimension
    (last mode), as we want to preserve client diversity.
    
    Args:
        W_tensor: Parameter tensor (*param_shape, num_clients)
        device: Device
        skip_last_mode: If True, skip mode p-1 (client dimension)
    
    Returns:
        Gradient tensor
    """
    W_tensor = W_tensor.to(device)
    grad = torch.zeros_like(W_tensor)
    p = W_tensor.dim()
    
    # Determine which modes to process
    modes = range(p - 1) if skip_last_mode else range(p)
    
    for k in modes:
        W_k = unfold(W_tensor, k)
        
        try:
            U, S, Vh = torch.linalg.svd(W_k, full_matrices=False)
            V = Vh.T
        except RuntimeError:
            W_k_reg = W_k + 1e-6 * torch.randn_like(W_k)
            U, S, Vh = torch.linalg.svd(W_k_reg, full_matrices=False)
            V = Vh.T
        
        grad_k_matrix = torch.mm(U, V.T)
        grad_k_tensor = fold(grad_k_matrix, k, W_tensor.shape)
        grad += grad_k_tensor
    
    return grad
