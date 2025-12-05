"""PACS dataset implementation using HuggingFace datasets.

PACS contains 4 domains: Photo, Art_painting, Cartoon, Sketch
with 7 classes: dog, elephant, giraffe, guitar, horse, house, person
"""

import os
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# HuggingFace datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from utils.transforms import build_transforms


# PACS domain and class mappings
PACS_DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
PACS_CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PACS_NUM_CLASSES = 7


class PACSDataset(Dataset):
    """PACS dataset for federated learning.

    Loads data from HuggingFace datasets library.
    """

    def __init__(
        self,
        root: str = None,
        indices: Optional[List[int]] = None,
        transform=None,
        train: bool = True,
        download: bool = True,
        cache_dir: Optional[str] = None
    ):
        """Initialize PACS dataset.

        Args:
            root: Root directory for caching (optional, uses HF cache if None)
            indices: Optional list of sample indices to use (for client partitioning)
            transform: Optional image transforms
            train: Whether this is training data
            download: Whether to download if not cached
            cache_dir: Optional cache directory for HuggingFace datasets
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library required. "
                "Install with: pip install datasets"
            )

        self.root = root
        self.train = train
        self.cache_dir = cache_dir or root

        # Load dataset from HuggingFace
        self._load_hf_dataset()

        # Filter by indices if provided
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        # Set transform - use default if not provided
        if transform is not None:
            self.transform = transform
        else:
            self.transform = build_transforms(train=train)

        self.domains = PACS_DOMAINS
        self.num_classes = PACS_NUM_CLASSES
        self.classes = PACS_CLASSES

    def _load_hf_dataset(self):
        """Load PACS dataset from HuggingFace.

        Note: PACS on HuggingFace only has 'train' split.
        Train/val split is handled by partition.py using indices.
        """
        # Always load from 'train' split (only available split)
        try:
            dataset = load_dataset(
                "flwrlabs/pacs",
                split='train',
                cache_dir=self.cache_dir
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PACS dataset from HuggingFace: {e}\n"
                "Make sure you have internet connection for first download."
            )

        # Convert to our internal format
        self.samples = []
        for idx, item in enumerate(dataset):
            self.samples.append({
                'image': item['image'],  # PIL Image
                'label': item['label'],
                'domain': PACS_DOMAINS[item['domain']]
                    if isinstance(item['domain'], int)
                    else item['domain']
            })

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
        image = sample['image']
        label = sample['label']
        domain = sample['domain']

        # Ensure image is PIL Image and RGB
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')

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

    def get_domain_distribution(self) -> Dict[str, int]:
        """Get domain distribution in the dataset.

        Returns:
            Dictionary mapping domain name to count
        """
        distribution = {}
        for sample in self.samples:
            domain = sample['domain']
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution

    def build_index(self) -> Dict:
        """Build index compatible with partition.py.

        Returns:
            Index dictionary with samples list
        """
        index = {
            'domains': self.domains,
            'num_classes': self.num_classes,
            'classes': self.classes,
            'samples': []
        }

        for i, sample in enumerate(self.samples):
            index['samples'].append({
                'idx': i,
                'label': sample['label'],
                'domain': sample['domain']
            })

        return index


def build_pacs_index(
    cache_dir: Optional[str] = None,
    train: bool = True
) -> Dict:
    """Build PACS dataset index for partitioning.

    Args:
        cache_dir: Cache directory for HuggingFace datasets
        train: Whether to use training split

    Returns:
        Index dictionary compatible with partition.py
    """
    dataset = PACSDataset(
        cache_dir=cache_dir,
        train=train,
        transform=None
    )
    return dataset.build_index()


def get_pacs_stats():
    """Print PACS dataset statistics."""
    print("Loading PACS dataset...")

    dataset = PACSDataset(train=True, transform=None)

    print(f"\n{'='*50}")
    print("PACS Dataset Statistics")
    print(f"{'='*50}")
    print(f"Domains: {PACS_DOMAINS}")
    print(f"Classes: {PACS_CLASSES}")
    print(f"Number of classes: {PACS_NUM_CLASSES}")

    print(f"\nTotal samples: {len(dataset)}")
    domain_dist = dataset.get_domain_distribution()
    for domain, count in sorted(domain_dist.items()):
        print(f"  {domain}: {count}")

    print(f"\nClass distribution:")
    class_dist = dataset.get_class_distribution()
    for class_id, count in sorted(class_dist.items()):
        print(f"  {PACS_CLASSES[class_id]}: {count}")


if __name__ == '__main__':
    get_pacs_stats()
