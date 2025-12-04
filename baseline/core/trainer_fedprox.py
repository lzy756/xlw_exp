"""FedProx local trainer for baseline experiments.

Extends FedAvg with proximal term: loss = CE + (μ/2)||w - w_t||²
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from baseline.core.trainer_fedavg import LocalTrainerFedAvg


class LocalTrainerFedProx(LocalTrainerFedAvg):
    """Local trainer for FedProx baseline.

    Adds proximal regularization to FedAvg:
    loss = CE + (μ/2)||w - w_t||²
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        mu: float = 0.01
    ):
        """Initialize FedProx local trainer.

        Args:
            model: Neural network model
            device: Device for training (cuda/cpu)
            lr: Learning rate for all parameters
            weight_decay: Weight decay for regularization
            mu: Proximal term coefficient (default 0.01)
        """
        super().__init__(model, device, lr, weight_decay)
        self.mu = mu

    def _compute_proximal_term(
        self,
        global_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute proximal term ||w - w_t||².

        Args:
            global_state: Global model state dictionary (w_t)

        Returns:
            Proximal loss value
        """
        prox_loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in global_state:
                # Move global param to device if needed
                global_param = global_state[name]
                if global_param.device != self.device:
                    global_param = global_param.to(self.device)

                # Accumulate squared L2 distance
                prox_loss = prox_loss + torch.sum((param - global_param) ** 2)

        return prox_loss

    def train_client(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        local_steps: int = 5,
        w_global_snapshot: Dict[str, torch.Tensor] | None = None
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Train model on client's local data with proximal term.

        Args:
            dataset: Client's local dataset
            batch_size: Batch size for training
            local_steps: Number of local training steps
            w_global_snapshot: Global model snapshot for proximal term

        Returns:
            Tuple of (state_dict, num_samples)
        """
        # If no global snapshot provided, fall back to FedAvg
        if w_global_snapshot is None or self.mu == 0:
            return super().train_client(dataset, batch_size, local_steps)

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

        # Local training with proximal term
        for step in range(local_steps):
            for batch in dataloader:
                # DomainNetDataset returns (image, label, domain)
                images, labels = batch[0], batch[1]
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    outputs = self.model(images)
                    ce_loss = self.criterion(outputs, labels)

                # Compute proximal term: (μ/2)||w - w_t||²
                prox_loss = self._compute_proximal_term(w_global_snapshot)
                total_loss = ce_loss + (self.mu / 2.0) * prox_loss

                # Backward pass
                scaler.scale(total_loss).backward()

                # Update parameters
                scaler.step(optimizer)
                scaler.update()

        # Extract state dictionary
        state_dict = self.model.state_dict_global()

        return state_dict, num_samples
