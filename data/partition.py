"""Data partitioning utilities for federated learning."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import random


def dirichlet_split_by_class(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    epsilon: float = 1e-6
) -> List[List[int]]:
    """Split data using Dirichlet distribution based on class labels.

    Args:
        labels: Array of class labels
        num_clients: Number of clients to split data among
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)
        seed: Random seed for reproducibility
        epsilon: Smoothing factor to avoid empty allocations

    Returns:
        List of lists, where each inner list contains sample indices for a client
    """
    np.random.seed(seed)
    random.seed(seed)

    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # Process each class
    for class_id in unique_labels:
        # Get indices for this class
        class_indices = np.where(labels == class_id)[0].tolist()
        n_class_samples = len(class_indices)

        if n_class_samples == 0:
            continue

        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)

        # Add epsilon smoothing to ensure minimum samples
        proportions = proportions + epsilon
        proportions = proportions / proportions.sum()

        # Calculate number of samples per client for this class
        client_counts = (proportions * n_class_samples).astype(int)

        # Adjust for rounding errors
        diff = n_class_samples - client_counts.sum()
        if diff > 0:
            # Add remaining samples to clients with highest proportions
            top_clients = np.argsort(proportions)[-diff:]
            for client_id in top_clients:
                client_counts[client_id] += 1
        elif diff < 0:
            # Remove samples from clients with lowest proportions
            bottom_clients = np.argsort(proportions)[:abs(diff)]
            for client_id in bottom_clients:
                if client_counts[client_id] > 0:
                    client_counts[client_id] -= 1

        # Randomly shuffle class indices
        random.shuffle(class_indices)

        # Assign samples to clients
        start_idx = 0
        for client_id, count in enumerate(client_counts):
            if count > 0:
                end_idx = start_idx + count
                client_indices[client_id].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx

    # Validate minimum sample requirement (≥5 samples per client)
    for client_id, indices in enumerate(client_indices):
        if len(indices) < 5:
            # Find clients with most samples and redistribute
            while len(indices) < 5:
                # Find client with most samples
                max_client = max(
                    range(num_clients),
                    key=lambda i: len(client_indices[i]) if i != client_id else 0
                )
                if len(client_indices[max_client]) <= 5:
                    break  # Cannot redistribute without violating minimum

                # Transfer one sample
                transfer_idx = client_indices[max_client].pop()
                client_indices[client_id].append(transfer_idx)

    return client_indices


def build_domain_clients(
    index: Dict,
    domain: str,
    num_clients: int,
    alpha: float,
    unload_ratio: float,
    val_ratio: float,
    seed: int
) -> Dict[int, Dict[str, List[int]]]:
    """Build client data partitions for a single domain.

    Args:
        index: Dataset index containing all samples
        domain: Domain name to partition
        num_clients: Number of clients in this domain
        alpha: Dirichlet concentration parameter
        unload_ratio: Fraction of data to offload to DC (ρ)
        val_ratio: Fraction of data for validation
        seed: Random seed

    Returns:
        Dictionary mapping client_id to dict with 'local', 'unload', 'val', 'dc_unload' indices
    """
    np.random.seed(seed)
    random.seed(seed)

    # Extract domain samples
    domain_samples = []
    domain_labels = []

    for i, sample in enumerate(index['samples']):
        if sample['domain'] == domain:
            domain_samples.append(i)
            domain_labels.append(sample['label'])

    domain_labels = np.array(domain_labels)

    # Perform Dirichlet split
    client_partitions = dirichlet_split_by_class(
        domain_labels,
        num_clients,
        alpha,
        seed=seed
    )

    # Build client dictionaries
    clients = {}
    dc_unload_pool = []

    for client_id, sample_indices in enumerate(client_partitions):
        if len(sample_indices) == 0:
            # Handle empty client (shouldn't happen with minimum sample validation)
            clients[client_id] = {
                'local': [],
                'unload': [],
                'val': [],
                'domain': domain
            }
            continue

        # Map back to global indices
        global_indices = [domain_samples[i] for i in sample_indices]
        random.shuffle(global_indices)

        # Split into train and validation
        n_samples = len(global_indices)
        n_val = max(1, int(n_samples * val_ratio))
        val_indices = global_indices[:n_val]
        train_indices = global_indices[n_val:]

        # Split training data into local and unload
        n_train = len(train_indices)
        n_unload = int(n_train * unload_ratio)
        unload_indices = train_indices[:n_unload]
        local_indices = train_indices[n_unload:]

        # Add to DC unload pool
        dc_unload_pool.extend(unload_indices)

        clients[client_id] = {
            'local': local_indices,
            'unload': unload_indices,
            'val': val_indices,
            'domain': domain
        }

    # Add DC unload pool to result
    return {
        'clients': clients,
        'dc_unload_pool': dc_unload_pool
    }