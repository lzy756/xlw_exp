"""FedAvg local trainer for baseline experiments.

Standard federated averaging with cross-entropy loss,
no proximal term or personalization.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class LocalTrainerFedAvg:
    """Local trainer for FedAvg baseline.

    Uses standard cross-entropy loss without any regularization.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr: float = 3e-4,
        weight_decay: float = 1e-4
    ):
        """Initialize FedAvg local trainer.

        Args:
            model: Neural network model
            device: Device for training (cuda/cpu)
            lr: Learning rate for all parameters
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

        # Move model to device
        self.model = self.model.to(device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def train_client(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        local_steps: int = 5
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Train model on client's local data.

        Args:
            dataset: Client's local dataset
            batch_size: Batch size for training
            local_steps: Number of local training steps (epochs over data)

        Returns:
            Tuple of (state_dict, num_samples)
        """
        self.model.train()

        num_samples = len(dataset)
        if num_samples == 0:
            return self.model.state_dict_global(), 0

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True
        )

        # Setup optimizer for all parameters
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        # Local training
        for step in range(local_steps):
            for batch in dataloader:
                # DomainNetDataset returns (image, label, domain)
                images, labels = batch[0], batch[1]
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass (domain ignored for single-head model)
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                scaler.scale(loss).backward()

                # Update parameters
                scaler.step(optimizer)
                scaler.update()

        # Extract state dictionary (move to CPU to save GPU memory)
        state_dict = self.model.state_dict_global()

        return state_dict, num_samples

    def evaluate(
        self,
        dataset: Dataset,
        batch_size: int = 32
    ) -> Tuple[float, float]:
        """Evaluate model on dataset.

        Args:
            dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation

        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        self.model.eval()

        if len(dataset) == 0:
            return 0.0, 0.0

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True
        )

        # Evaluation metrics
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch[0], batch[1]
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Track metrics
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        # Calculate metrics
        val_loss = total_loss / total if total > 0 else 0.0
        val_acc = 100.0 * correct / total if total > 0 else 0.0

        return val_loss, val_acc

    def load_global_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load global model state.

        Args:
            state_dict: State dictionary to load
        """
        self.model.load_state_dict_global(state_dict)
