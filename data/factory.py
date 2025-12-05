"""Dataset factory for unified data loading.

Supports both DomainNet and PACS datasets through a common interface.
"""

from typing import Dict, List, Optional, Tuple
import json
import os

from torch.utils.data import Dataset

from data.domainnet import DomainNetDataset
from data.pacs import PACSDataset, build_pacs_index, PACS_DOMAINS, PACS_NUM_CLASSES
from data.partition import build_domain_clients


def get_dataset_class(dataset_name: str):
    """Get dataset class by name.

    Args:
        dataset_name: 'domainnet' or 'pacs'

    Returns:
        Dataset class
    """
    if dataset_name.lower() == 'pacs':
        return PACSDataset
    else:
        return DomainNetDataset


def create_dataset(
    config: Dict,
    indices: Optional[List[int]] = None,
    train: bool = True
) -> Dataset:
    """Create dataset instance based on config.

    Args:
        config: Configuration dictionary with 'data' section
        indices: Optional sample indices for partitioning
        train: Whether this is training data

    Returns:
        Dataset instance
    """
    dataset_name = config['data'].get('dataset', 'domainnet')

    if dataset_name.lower() == 'pacs':
        return PACSDataset(
            root=config['data'].get('root'),
            indices=indices,
            train=train,
            cache_dir=config['data'].get('cache_dir')
        )
    else:
        return DomainNetDataset(
            root=config['data']['root'],
            indices=indices,
            train=train
        )


def build_index(config: Dict, train: bool = True) -> Dict:
    """Build or load dataset index for partitioning.

    Args:
        config: Configuration dictionary
        train: Whether to use training split

    Returns:
        Index dictionary with 'samples', 'domains', 'num_classes'
    """
    dataset_name = config['data'].get('dataset', 'domainnet')

    if dataset_name.lower() == 'pacs':
        # Build index from PACS dataset
        return build_pacs_index(
            cache_dir=config['data'].get('root'),
            train=train
        )
    else:
        # Load index from DomainNet
        index_path = os.path.join(config['data']['root'], 'index.json')

        if not os.path.exists(index_path):
            # Create dummy index for testing
            _ = DomainNetDataset(config['data']['root'])

        with open(index_path, 'r') as f:
            return json.load(f)


def prepare_federated_data(config: Dict) -> Tuple[Dict, Dict]:
    """Prepare federated training and validation data.

    This is the main entry point for data preparation in federated setting.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_data, val_data) where:
            - train_data[domain] contains 'clients' and 'dc_unload_pool'
            - val_data[domain] is a Dataset instance
    """
    dataset_name = config['data'].get('dataset', 'domainnet')

    # Build index
    index = build_index(config, train=True)

    train_data = {}
    val_data = {}

    for domain_idx, domain in enumerate(config['data']['domains']):
        # Build client partitions for this domain
        domain_data = build_domain_clients(
            index=index,
            domain=domain,
            num_clients=config['partition']['num_clients_per_domain'],
            alpha=config['partition']['alpha'],
            unload_ratio=config['partition']['unload_ratio'],
            val_ratio=config['partition']['val_ratio'],
            seed=config['partition']['seed'] + domain_idx
        )

        train_data[domain] = domain_data

        # Aggregate validation indices from all clients
        val_indices = []
        for client_data in domain_data['clients'].values():
            val_indices.extend(client_data['val'])

        # Create validation dataset
        val_data[domain] = create_dataset(
            config=config,
            indices=val_indices,
            train=False
        )

    return train_data, val_data


def get_dataset_info(config: Dict) -> Dict:
    """Get dataset information.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with 'domains', 'num_classes', 'name'
    """
    dataset_name = config['data'].get('dataset', 'domainnet')

    if dataset_name.lower() == 'pacs':
        return {
            'name': 'PACS',
            'domains': PACS_DOMAINS,
            'num_classes': PACS_NUM_CLASSES
        }
    else:
        return {
            'name': 'DomainNet',
            'domains': config['data'].get('domains', [
                'clipart', 'infograph', 'painting',
                'quickdraw', 'real', 'sketch'
            ]),
            'num_classes': config['data'].get('num_classes', 126)
        }
