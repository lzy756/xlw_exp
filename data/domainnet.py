"""DomainNet dataset implementation."""

import json
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.transforms import build_transforms


class DomainNetDataset(Dataset):
    """DomainNet dataset for federated learning.

    Loads samples from index.json which contains paths and labels.
    """

    def __init__(
        self,
        root: str,
        indices: Optional[List[int]] = None,
        transform=None,
        train: bool = True
    ):
        """Initialize DomainNet dataset.

        Args:
            root: Root directory containing DomainNet data
            indices: Optional list of sample indices to use (for client partitioning)
            transform: Optional image transforms
            train: Whether this is training data (affects default transforms)
        """
        self.root = root
        self.train = train

        # Load index file
        index_path = os.path.join(root, 'index.json')
        if not os.path.exists(index_path):
            # Create a dummy index for testing if it doesn't exist
            self._create_dummy_index(index_path)

        with open(index_path, 'r') as f:
            self.index = json.load(f)

        # Filter samples if indices provided
        if indices is not None:
            self.samples = [self.index['samples'][i] for i in indices]
        else:
            self.samples = self.index['samples']

        # Set transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = build_transforms(train=train)

        # Extract metadata
        self.domains = self.index.get('domains', [])
        self.num_classes = self.index.get('num_classes', 126)

    def _create_dummy_index(self, index_path: str) -> None:
        """Create a dummy index file for testing purposes.

        Args:
            index_path: Path to save index.json
        """
        dummy_index = {
            'domains': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
            'num_classes': 126,
            'samples': []
        }

        # Create some dummy samples
        for domain in dummy_index['domains']:
            for class_id in range(10):  # Just 10 classes for dummy
                for sample_id in range(5):  # 5 samples per class
                    dummy_index['samples'].append({
                        'path': f'{domain}/class_{class_id}/sample_{sample_id}.jpg',
                        'label': class_id,
                        'domain': domain
                    })

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, 'w') as f:
            json.dump(dummy_index, f, indent=2)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label, domain)
        """
        sample = self.samples[idx]
        path = os.path.join(self.root, sample['path'])
        label = sample['label']
        domain = sample['domain']

        # Load image
        if os.path.exists(path):
            image = Image.open(path).convert('RGB')
        else:
            # Create dummy image if file doesn't exist (for testing)
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label, domain

    def get_domain_samples(self, domain: str) -> List[int]:
        """Get indices of samples belonging to a specific domain.

        Args:
            domain: Domain name

        Returns:
            List of sample indices for the domain
        """
        indices = []
        for i, sample in enumerate(self.samples):
            if sample['domain'] == domain:
                indices.append(i)
        return indices

    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution in the dataset.

        Returns:
            Dictionary mapping class_id to count
        """
        distribution = {}
        for sample in self.samples:
            label = sample['label']
            distribution[label] = distribution.get(label, 0) + 1
        return distribution