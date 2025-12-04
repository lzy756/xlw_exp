"""PACS dataset utilities for FL setup.

Domains: photo, art_painting, cartoon, sketch
Assumes an index.json with entries: {"path": str, "label": int, "domain": str}
If missing, will create a dummy index for quick tests.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]


def build_transforms(img_size: int = 160, train: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


class PACSDataset(Dataset):
    def __init__(self, root: str, indices: Optional[List[int]] = None, train: bool = True, img_size: int = 160):
        self.root = root
        index_path = Path(root) / 'index.json'
        if not index_path.exists():
            self._create_dummy_index(index_path)
        with open(index_path, 'r') as f:
            self.index = json.load(f)
        if indices is not None:
            self.samples = [self.index[i] for i in indices]
        else:
            self.samples = self.index
        self.transform = build_transforms(img_size=img_size, train=train)

    def _create_dummy_index(self, index_path: Path):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        samples = []
        for d in PACS_DOMAINS:
            for c in range(7):
                for i in range(5):
                    samples.append({
                        "path": f"{d}/class_{c}/sample_{i}.jpg",
                        "label": c,
                        "domain": d
                    })
        with open(index_path, 'w') as f:
            json.dump(samples, f, indent=2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.samples[idx]
        path = Path(self.root) / item['path']
        label = item['label']
        domain = item['domain']
        if path.exists():
            img = Image.open(path).convert('RGB')
        else:
            img = Image.new('RGB', (160, 160), color=(128, 128, 128))
        img = self.transform(img)
        return img, label, domain


# Simple partitioning: stratified by class within domain
import numpy as np
import random

def dirichlet_split(labels: np.ndarray, num_clients: int, alpha: float, seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    clients = [[] for _ in range(num_clients)]
    classes = np.unique(labels)
    for c in classes:
        idxs = np.where(labels == c)[0].tolist()
        proportions = np.random.dirichlet([alpha]*num_clients)
        proportions = (proportions + 1e-6) / (proportions.sum() + 1e-6*len(proportions))
        counts = (proportions * len(idxs)).astype(int)
        diff = len(idxs) - counts.sum()
        for i in np.argsort(proportions)[-diff:]:
            counts[i] += 1
        random.shuffle(idxs)
        start = 0
        for i, cnt in enumerate(counts):
            clients[i].extend(idxs[start:start+cnt])
            start += cnt
    return clients

def build_domain_clients(index: List[Dict], domain: str, num_clients: int, alpha: float, unload_ratio: float, val_ratio: float, seed: int):
    domain_idxs = [i for i, s in enumerate(index) if s['domain'] == domain]
    labels = np.array([index[i]['label'] for i in domain_idxs])
    splits = dirichlet_split(labels, num_clients, alpha, seed)
    clients = {}
    dc_unload_pool = []
    for cid, local_rel in enumerate(splits):
        local = [domain_idxs[i] for i in local_rel]
        random.shuffle(local)
        n_val = max(1, int(len(local) * val_ratio))
        val = local[:n_val]
        train_idxs = local[n_val:]
        n_unload = int(len(train_idxs) * unload_ratio)
        unload = train_idxs[:n_unload]
        local_train = train_idxs[n_unload:]
        dc_unload_pool.extend(unload)
        clients[cid] = {
            'local': local_train,
            'unload': unload,
            'val': val,
            'domain': domain
        }
    return {'clients': clients, 'dc_unload_pool': dc_unload_pool}
