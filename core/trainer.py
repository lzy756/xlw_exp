"""Local trainer for federated learning clients."""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR


class LocalTrainer:
    """Handles local training for clients and evaluation for domains."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr_theta: float = 3e-4,
        lr_phi: float = 1e-3,
        weight_decay: float = 1e-4,
        cosine_lr: bool = True
    ):
        """Initialize local trainer.

        Args:
            model: Neural network model with parameters_theta/phi methods
            device: Device for training (cuda/cpu)
            lr_theta: Learning rate for backbone parameters
            lr_phi: Learning rate for domain-specific parameters
            weight_decay: Weight decay for regularization
            cosine_lr: Whether to use cosine learning rate schedule
        """
        self.model = model
        self.device = device
        self.lr_theta = lr_theta
        self.lr_phi = lr_phi
        self.weight_decay = weight_decay
        self.cosine_lr = cosine_lr

        # Move model to device
        self.model = self.model.to(device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def train_client(
        self,
        domain: str,
        dataset: Dataset,
        batch_size: int = 32,
        local_steps: int = 5,
        lr_phi: Optional[float] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], float]:
        """Train model on client's local data.

        Args:
            domain: Domain name for this client
            dataset: Client's local dataset
            batch_size: Batch size for training
            local_steps: Number of local training steps
            lr_phi: Optional override for phi learning rate

        Returns:
            Tuple of (theta_state_dict, phi_state_dict, train_accuracy)
        """
        self.model.train()

        # Use provided lr_phi or default
        if lr_phi is None:
            lr_phi = self.lr_phi

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )

        # Setup dual optimizers
        theta_params = self.model.parameters_theta()
        phi_params = self.model.parameters_phi(domain)

        optimizer_theta = optim.Adam(
            theta_params,
            lr=self.lr_theta,
            weight_decay=self.weight_decay
        )

        optimizer_phi = optim.Adam(
            phi_params,
            lr=lr_phi,
            weight_decay=self.weight_decay
        )

        # Setup schedulers if using cosine LR
        if self.cosine_lr:
            total_iters = local_steps * len(dataloader)
            scheduler_theta = CosineAnnealingLR(optimizer_theta, T_max=total_iters)
            scheduler_phi = CosineAnnealingLR(optimizer_phi, T_max=total_iters)

        # Training metrics
        total_loss = 0
        correct = 0
        total = 0

        # Local training
        for step in range(local_steps):
            for batch_idx, (images, labels, domains) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                optimizer_theta.zero_grad()
                optimizer_phi.zero_grad()

                # Forward pass
                outputs = self.model(images, domain)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer_theta.step()
                optimizer_phi.step()

                # Update schedulers
                if self.cosine_lr:
                    scheduler_theta.step()
                    scheduler_phi.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        # Calculate accuracy
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Extract state dictionaries (move to CPU to save GPU memory)
        theta_state = {
            k: v.cpu().clone()
            for k, v in self.model.state_dict().items()
            if 'heads.' not in k and 'lora_blocks.' not in k
        }

        phi_state = {
            k: v.cpu().clone()
            for k, v in self.model.state_dict().items()
            if 'lora_blocks.' in k or f'heads.{domain}' in k
        }

        return theta_state, phi_state, train_acc

    def evaluate_domain(
        self,
        domain: str,
        dataset: Dataset,
        batch_size: int = 32,
        edge_manager: Optional[object] = None
    ) -> Tuple[float, float, float]:
        """Evaluate model on domain's validation data.

        Args:
            domain: Domain name
            dataset: Validation dataset
            batch_size: Batch size for evaluation
            edge_manager: Optional EdgeManager for updating eval stats

        Returns:
            Tuple of (val_loss, val_accuracy, L_e)
        """
        self.model.eval()

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )

        # Evaluation metrics
        total_loss = 0
        correct = 0
        total = 0
        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels, domains in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Extract features if edge_manager provided
                if edge_manager is not None:
                    features = self.model.forward_features(images)
                    all_features.append(features.cpu())
                    all_labels.append(labels.cpu())

                # Forward pass
                outputs = self.model(images, domain)
                loss = self.criterion(outputs, labels)

                # Track metrics
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        # Calculate metrics
        val_loss = total_loss / total if total > 0 else 0.0
        val_acc = 100.0 * correct / total if total > 0 else 0.0

        # Update edge manager if provided
        L_e = val_loss  # Default to val_loss
        if edge_manager is not None and all_features:
            # Concatenate all features and labels
            features = torch.cat(all_features, dim=0).to(self.device)
            labels = torch.cat(all_labels, dim=0).to(self.device)

            # Update edge manager stats
            L_e = edge_manager.update_eval_stats(
                domain=domain,
                val_loss=val_loss,
                feats=features,
                labels=labels
            )

        return val_loss, val_acc, L_e


def dc_train_on_offload_pool(
    model: nn.Module,
    domain: str,
    dataset: Dataset,
    batch_size: int = 32,
    device: str = 'cuda',
    lr: float = 1e-3,
    steps: int = 3
) -> Dict[str, torch.Tensor]:
    """Optional training on DC's offload pool.

    Args:
        model: Model to train
        domain: Domain name
        dataset: Offload pool dataset
        batch_size: Batch size
        device: Device for training
        lr: Learning rate
        steps: Number of training steps

    Returns:
        Updated phi state dict for the domain
    """
    model.train()
    model = model.to(device)

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )

    # Setup optimizer for phi parameters only
    phi_params = model.parameters_phi(domain)
    optimizer = optim.Adam(phi_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for step in range(steps):
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, domain)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Extract phi state dict
    phi_state = {
        k: v.cpu().clone()
        for k, v in model.state_dict().items()
        if 'lora_blocks.' in k or f'heads.{domain}' in k
    }

    return phi_state