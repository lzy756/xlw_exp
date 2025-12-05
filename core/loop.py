"""V2 federated learning training loop.

Key differences from v1:
1. θ (backbone) aggregated every round (not just every K rounds)
2. No LoRA - φ is only domain heads
3. No calibration step (disabled in Phase 1)
4. No DC training (disabled in Phase 1)
5. Optional fair-weighting every K rounds
"""

import os
import random
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.factory import create_dataset
from core.aggregator import (
    aggregate_theta,
    aggregate_phi_domain,
    fair_weighted_aggregate
)
from core.edge_manager import EdgeManager
from core.selector import FAPSelector
from utils.metrics import per_domain_metrics
from utils.common import get_git_commit_hash


class LocalTrainer:
    """Simplified local trainer for v2 models (no LoRA)."""

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
        dataset,
        batch_size: int = 32,
        local_steps: int = 5,
        lr_phi: Optional[float] = None
    ):
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
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True
        )

        # Setup dual optimizers
        theta_params = self.model.parameters_theta()
        phi_params = self.model.parameters_phi(domain)

        optimizer_theta = torch.optim.Adam(
            theta_params,
            lr=self.lr_theta,
            weight_decay=self.weight_decay
        )

        optimizer_phi = torch.optim.Adam(
            phi_params,
            lr=lr_phi,
            weight_decay=self.weight_decay
        )

        # Setup schedulers if using cosine LR
        if self.cosine_lr:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            total_iters = local_steps * len(dataloader)
            scheduler_theta = CosineAnnealingLR(optimizer_theta, T_max=max(1, total_iters))
            scheduler_phi = CosineAnnealingLR(optimizer_phi, T_max=max(1, total_iters))
        else:
            scheduler_theta = scheduler_phi = None

        # AMP scaler (shared for both optimizers)
        use_amp = self.device == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Training metrics
        correct = 0
        total = 0

        # Track iteration for scheduler
        iteration = 0

        # Local training
        for step in range(local_steps):
            for batch_idx, (images, labels, domains) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                optimizer_theta.zero_grad()
                optimizer_phi.zero_grad()

                # Forward pass with AMP
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(images, domain)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                scaler.scale(loss).backward()

                # Update parameters
                scaler.step(optimizer_theta)
                scaler.step(optimizer_phi)
                scaler.update()

                # Track metrics
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                iteration += 1

        # Update schedulers AFTER all training steps (per-client)
        if self.cosine_lr and scheduler_theta is not None and iteration > 0:
            for _ in range(iteration):
                scheduler_theta.step()
                scheduler_phi.step()

        # Calculate accuracy
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Extract state dictionaries (v2: no lora_blocks)
        theta_state = {
            k: v.cpu().clone()
            for k, v in self.model.state_dict().items()
            if not k.startswith('biases.') and not k.startswith('adapters_')
        }

        phi_state = {
            k: v.cpu().clone()
            for k, v in self.model.state_dict().items()
            if k == f'biases.{domain}'
            or k.startswith(f'adapters_down.{domain}')
            or k.startswith(f'adapters_up.{domain}')
        }

        return theta_state, phi_state, train_acc

    def evaluate_domain(
        self,
        domain: str,
        dataset,
        batch_size: int = 32,
        edge_manager: Optional[EdgeManager] = None
    ):
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
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True
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
                with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
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


def run_training(
    config: Dict,
    model: nn.Module,
    train_data: Dict[str, Dict],
    val_data: Dict[str, Dataset],
    edge_manager: EdgeManager,
    selector: FAPSelector,
    trainer: LocalTrainer,
    logger,
    exp_dir: str
) -> Dict:
    """Run v2 federated learning training loop.
    
    Key changes from v1:
    1. θ aggregated every round (not just every K rounds)
    2. Optional fair-weighting every K rounds
    3. No calibration or DC training (Phase 1)

    Args:
        config: Configuration dictionary
        model: Neural network model (v2 model, no LoRA)
        train_data: Training data per domain and client
        val_data: Validation data per domain
        edge_manager: EdgeManager instance
        selector: FAPSelectorV2 instance
        trainer: LocalTrainerV2 instance
        logger: Logger instance
        exp_dir: Experiment directory path

    Returns:
        Dictionary with training metrics
    """
    # Extract config parameters
    total_rounds = config['training']['total_rounds']
    local_steps = config['training']['local_steps']
    batch_size = config['training']['batch_size']
    clients_participation = config['training']['clients_participation']
    K = config['selector']['K']
    checkpoint_interval = config['logging']['checkpoint_interval']
    domains = config['data']['domains']
    
    # Fair-weighting settings
    fair_weighting_enabled = config.get('fair_weighting', {}).get('enable', False)
    
    # Use the provided experiment directory
    output_dir = exp_dir

    # Initialize global theta (backbone parameters, no LoRA)
    theta_global = {
        k: v.cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith('biases.')
    }

    # Track metrics
    metrics_history = {
        'avg_acc': [],
        'worst_acc': [],
        'variance': [],
        'per_domain_acc': {d: [] for d in domains},
        'fair_factors': [],  # Fairness factors (q)
    }

    logger.info(f"Starting v2 federated learning training for {total_rounds} rounds")
    logger.info(f"Configuration: K={K}, participation={clients_participation}")
    logger.info(f"Fair-weighting: {'enabled' if fair_weighting_enabled else 'disabled'}")
    logger.info(f"Git commit: {get_git_commit_hash()}")

    # Main training loop
    for round_num in range(1, total_rounds + 1):
        logger.info('='*50)
        logger.info(f"Round {round_num}/{total_rounds}")
        logger.info('='*50)

        # Update coverage at round start
        edge_manager.begin_round()

        # Phase 1: Client Sampling
        participating_clients = {}
        for domain in domains:
            domain_clients = train_data[domain]['clients']
            num_clients = len(domain_clients)
            num_selected = max(1, int(num_clients * clients_participation))

            # Random sampling
            selected_ids = random.sample(range(num_clients), num_selected)
            participating_clients[domain] = selected_ids

            logger.info(f"Domain {domain}: selected {len(selected_ids)} clients")

        # Phase 2: Local Training
        all_theta_updates = []
        all_theta_weights = []
        domain_theta_updates = {d: [] for d in domains}
        domain_theta_weights = {d: [] for d in domains}
        domain_phi_updates = {d: [] for d in domains}
        domain_phi_weights = {d: [] for d in domains}

        for domain in domains:
            # Get domain's current phi (will be loaded for each client)
            domain_phi_state = edge_manager.get_phi(domain)

            for client_id in participating_clients[domain]:
                # IMPORTANT: Load global theta BEFORE each client's training
                # This ensures each client starts from the same global model
                model.load_state_dict(theta_global, strict=False)
                
                # Load domain's current phi
                if domain_phi_state is not None:
                    model.load_state_dict(domain_phi_state, strict=False)

                # Get client's dataset
                client_data = train_data[domain]['clients'][client_id]
                client_dataset = create_dataset(
                    config=config,
                    indices=client_data['local'],
                    train=True
                )

                if len(client_dataset) == 0:
                    logger.warning(f"Client {client_id} in {domain} has no data, skipping")
                    continue

                # Local training
                theta_state, phi_state, train_acc = trainer.train_client(
                    domain=domain,
                    dataset=client_dataset,
                    batch_size=batch_size,
                    local_steps=local_steps
                )

                # Collect updates (both flat and per-domain)
                all_theta_updates.append(theta_state)
                all_theta_weights.append(len(client_dataset))
                domain_theta_updates[domain].append(theta_state)
                domain_theta_weights[domain].append(len(client_dataset))
                domain_phi_updates[domain].append(phi_state)
                domain_phi_weights[domain].append(len(client_dataset))

                # logger.info(f"  Client {client_id}: train_acc={train_acc:.2f}%")

        # Phase 3: Domain-Internal Phi Aggregation
        for domain in domains:
            if domain_phi_updates[domain]:
                aggregated_phi = aggregate_phi_domain(
                    client_phi_list=domain_phi_updates[domain],
                    client_weights=domain_phi_weights[domain],
                    domain=domain
                )
                edge_manager.set_phi(domain, aggregated_phi)
                logger.info(f"Aggregated phi for domain {domain}")

        # Phase 4: Global θ Aggregation (EVERY ROUND! Key v2 change)
        if all_theta_updates:
            # Check if this is a fair-weighting round
            use_fair_weighting = (round_num % K == 0) and fair_weighting_enabled
            
            if use_fair_weighting:
                logger.info(f"Round {round_num}: Applying fair-weighted aggregation")
                
                # Get metrics for fair factors
                L_map = {d: edge_manager.L_ema[d].get() or 0.0 for d in domains}
                drift_map = edge_manager.get_drift_scores()
                
                logger.info(f"Domain EMA losses (L_e): {L_map}")
                logger.info(f"Domain drift scores (Δ_e): {drift_map}")
                
                # Compute fair factors
                fair_factors = selector.compute_fair_factors(L_map, drift_map)
                logger.info(f"Fair factors: {fair_factors}")
                
                # Fair-weighted aggregation
                theta_global = fair_weighted_aggregate(
                    domain_theta_updates=domain_theta_updates,
                    domain_theta_weights=domain_theta_weights,
                    domains=domains,
                    fair_factors=fair_factors
                )
                
                metrics_history['fair_factors'].append(fair_factors)
            else:
                # Standard FedAvg aggregation
                theta_global = aggregate_theta(
                    client_theta_list=all_theta_updates,
                    client_weights=all_theta_weights
                )
                
                metrics_history['fair_factors'].append(None)
            
            logger.info(f"Global theta aggregated (round {round_num})")

        # Phase 5: Evaluation
        domain_accuracies = {}
        for domain in domains:
            # Load global theta
            model.load_state_dict(theta_global, strict=False)
            
            # Load domain's aggregated phi
            phi_state = edge_manager.get_phi(domain)
            if phi_state is not None:
                model.load_state_dict(phi_state, strict=False)

            # Evaluate on domain's validation set
            val_loss, val_acc, L_e = trainer.evaluate_domain(
                domain=domain,
                dataset=val_data[domain],
                batch_size=batch_size,
                edge_manager=edge_manager
            )

            domain_accuracies[domain] = val_acc
            logger.info(f"Domain {domain}: val_acc={val_acc:.2f}%, L_e={L_e:.4f}")

        # Calculate aggregate metrics
        avg_acc, worst_acc, variance = per_domain_metrics(domain_accuracies)
        logger.info(f"Round {round_num} metrics:")
        logger.info(f"  Average accuracy: {avg_acc:.2f}%")
        logger.info(f"  Worst accuracy: {worst_acc:.2f}%")
        logger.info(f"  Variance: {variance:.4f}")

        # Store metrics
        metrics_history['avg_acc'].append(avg_acc)
        metrics_history['worst_acc'].append(worst_acc)
        metrics_history['variance'].append(variance)
        for domain in domains:
            metrics_history['per_domain_acc'][domain].append(domain_accuracies[domain])

        # Compute drift for next round (also resets prototype accumulators)
        for domain in domains:
            edge_manager.compute_drift(domain)

        # Phase 6: Checkpointing
        if round_num % checkpoint_interval == 0:
            # Save global theta
            theta_path = os.path.join(
                output_dir, 'checkpoints', f'theta_global_r{round_num}.pt'
            )
            torch.save({'state_dict': theta_global, 'round': round_num}, theta_path)
            logger.info(f"Saved theta checkpoint: {theta_path}")

            # Save domain phi states
            for domain in domains:
                phi_state = edge_manager.get_phi(domain)
                if phi_state is not None:
                    phi_path = os.path.join(
                        output_dir, 'checkpoints', f'phi_{domain}_r{round_num}.pt'
                    )
                    torch.save({'state_dict': phi_state, 'round': round_num}, phi_path)

    # Final checkpoint
    logger.info("Training completed! Saving final checkpoints...")

    # Save final theta
    theta_path = os.path.join(output_dir, 'checkpoints', 'theta_global_final.pt')
    torch.save({
        'state_dict': theta_global,
        'round': total_rounds,
        'git_commit': get_git_commit_hash()
    }, theta_path)

    # Save final phi states
    for domain in domains:
        phi_state = edge_manager.get_phi(domain)
        if phi_state is not None:
            phi_path = os.path.join(output_dir, 'checkpoints', f'phi_{domain}_final.pt')
            torch.save({
                'state_dict': phi_state,
                'round': total_rounds
            }, phi_path)

    # Save edge manager state
    edge_path = os.path.join(output_dir, 'checkpoints', 'edge_manager_final.pt')
    torch.save({
        'coverage': edge_manager.coverage,
        'L_ema': {d: ema.get() for d, ema in edge_manager.L_ema.items()},
        'prev_proto': edge_manager.prev_proto,
        'last_aggregator': edge_manager.last_aggregator
    }, edge_path)

    logger.info(f"All checkpoints saved to {output_dir}/checkpoints/")

    return metrics_history
