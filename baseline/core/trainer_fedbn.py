"""FedBN local trainer for baseline experiments.

Same training as FedAvg, but aggregation/load use FedBN state dicts
(non-BN parameters aggregated, BN kept local).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class LocalTrainerFedBN:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr: float = 3e-4,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def train_client(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        local_steps: int = 5
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        self.model.train()
        num_samples = len(dataset)
        if num_samples == 0:
            return self.model.state_dict_fedbn(), 0

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        for _ in range(local_steps):
            for images, labels, *_ in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        return self.model.state_dict_fedbn(), num_samples

    def evaluate(self, dataset: Dataset, batch_size: int = 32) -> Tuple[float, float]:
        self.model.eval()
        if len(dataset) == 0:
            return 0.0, 0.0

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True
        )

        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, *_ in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        val_loss = total_loss / total if total > 0 else 0.0
        val_acc = 100.0 * correct / total if total > 0 else 0.0
        return val_loss, val_acc

    def load_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict_fedbn(state_dict)
