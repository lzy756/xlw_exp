"""FedRep-style training loop with LoRA personalization.

Simplified main method: global backbone (θ) is aggregated every round via
FedAvg, while domain-specific LoRA adapters (φ) stay local (aggregated only
within domain if multiple clients are sampled). Fair-weighting/DC/other
mix-ins are removed to keep the method focused and stable.
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
    aggregate_phi_domain
)
from core.edge_manager import EdgeManager
from core.selector import FAPSelector
from core.fair_aggregation import (
    DomainStateManager,
    DriftCalculator,
    CloudAggregator
)
from utils.metrics import per_domain_metrics
from utils.common import get_git_commit_hash


class LocalTrainer:
    """Local trainer implementing FedRep-style alternating updates."""

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
        lr_phi: Optional[float] = None,
        decoupled_training: bool = True,
        phi_steps_ratio: float = 0.6
    ):
        """Train model on client's local data with FedRep-style decoupled training.

        Args:
            domain: Domain name for this client
            dataset: Client's local dataset
            batch_size: Batch size for training
            local_steps: Number of local training steps (epochs)
            lr_phi: Optional override for phi learning rate
            decoupled_training: Whether to use FedRep-style alternating training
            phi_steps_ratio: Ratio of steps for phi training (default 0.6 = 60%)

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

        # Calculate split point for decoupled training
        # FedRep style: first train phi (head), then train theta (body)
        phi_steps = int(local_steps * phi_steps_ratio) if decoupled_training else 0
        theta_steps = local_steps - phi_steps if decoupled_training else local_steps

        # Local training with decoupled phases
        for step in range(local_steps):
            # Determine training phase
            if decoupled_training:
                if step < phi_steps:
                    # Phase A: Train phi (LoRA), freeze theta
                    train_theta = False
                    train_phi = True
                else:
                    # Phase B: Train theta (backbone), freeze phi
                    train_theta = True
                    train_phi = False
            else:
                # Joint training (original behavior)
                train_theta = True
                train_phi = True

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

                # Gradient clipping to prevent explosion
                max_norm = 5.0
                if train_theta:
                    scaler.unscale_(optimizer_theta)
                    torch.nn.utils.clip_grad_norm_(theta_params, max_norm)
                if train_phi:
                    scaler.unscale_(optimizer_phi)
                    torch.nn.utils.clip_grad_norm_(phi_params, max_norm)

                # Update parameters based on training phase
                if train_theta:
                    scaler.step(optimizer_theta)
                if train_phi:
                    scaler.step(optimizer_phi)
                scaler.update()

                # Update schedulers after optimizer step
                if self.cosine_lr:
                    if train_theta and scheduler_theta is not None:
                        scheduler_theta.step()
                    if train_phi and scheduler_phi is not None:
                        scheduler_phi.step()

                # Track metrics
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                iteration += 1

        # Calculate accuracy
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # Extract state dictionaries using model's methods
        theta_state = self.model.state_dict_theta()
        phi_state = self.model.state_dict_phi(domain)

        return theta_state, phi_state, train_acc

    def evaluate_domain(
        self,
        domain: str,
        dataset,
        batch_size: int = 32,
        edge_manager: Optional[EdgeManager] = None,
        return_features: bool = False
    ) -> tuple:
        """Evaluate model on domain's validation data.

        Args:
            domain: Domain name
            dataset: Validation dataset
            batch_size: Batch size for evaluation
            edge_manager: Optional EdgeManager for updating eval stats
            return_features: If True, return (val_loss, val_acc, L_e, features, labels)

        Returns:
            Tuple of (val_loss, val_accuracy, L_e) or
            (val_loss, val_accuracy, L_e, features, labels) if return_features=True
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

        # Always collect features if edge_manager provided OR return_features requested
        collect_features = edge_manager is not None or return_features

        with torch.no_grad():
            for images, labels, domains in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Extract features if needed
                with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
                    if collect_features:
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

        # Return with features if requested
        if return_features and all_features:
            features_out = torch.cat(all_features, dim=0)
            labels_out = torch.cat(all_labels, dim=0)
            return val_loss, val_acc, L_e, features_out, labels_out

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
    
    FedRep simplification:
    1. θ aggregated every round (FedAvg)
    2. φ = domain LoRA kept local (optionally averaged within domain)
    3. Fair-weighting/DC/calibration removed

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
    checkpoint_interval = config['logging']['checkpoint_interval']
    domains = config['data']['domains']

    # FedRep decoupled training settings
    decoupled_training = config['training'].get('decoupled_training', True)
    phi_steps_ratio = config['training'].get('phi_steps_ratio', 0.6)

    # ========== FAP v3 Fair Aggregation Module (Optional) ==========
    # Initialize fair aggregation components if enabled
    fair_agg_config = config.get('fair_aggregation', {})
    use_fair_aggregation = fair_agg_config.get('enable', False)

    fair_state_mgr = None
    fair_drift_calc = None
    fair_cloud_agg = None

    if use_fair_aggregation:
        logger.info("[FAP-v3] Initializing fair aggregation module...")

        # Get feature dimension from model
        feature_dim = getattr(model, 'feature_dim', 512)
        num_classes = config['data'].get('num_classes', 126)

        # Initialize domain state manager (A1, A2, A4)
        fair_state_mgr = DomainStateManager(
            domains=domains,
            alpha_L=fair_agg_config.get('alpha_L', 0.9),
            alpha_min=fair_agg_config.get('alpha_min', 0.125),
            T_H=fair_agg_config.get('T_H', 15.0)
        )

        # Initialize drift calculator (A3)
        fair_drift_calc = DriftCalculator(
            feature_dim=feature_dim,
            proj_dim=config.get('edge_manager', {}).get('proj_dim', 64),
            num_classes=num_classes,
            tau_d=fair_agg_config.get('tau_d', 5),
            seed=config.get('system', {}).get('seed', 42),
            device=config.get('system', {}).get('device', 'cuda')
        )

        # Initialize cloud aggregator (B2-B5)
        fair_cloud_agg = CloudAggregator(
            domains=domains,
            w1=fair_agg_config.get('w1', 1.0),
            w2=fair_agg_config.get('w2', 0.3),
            w3=fair_agg_config.get('w3', 0.5),
            tau=fair_agg_config.get('tau', 1.0),
            gamma=fair_agg_config.get('gamma', 0.5),
            alpha_floor=fair_agg_config.get('alpha_floor', 0.05),
            alpha_ceil=fair_agg_config.get('alpha_ceil', 0.7),
            ema_beta=fair_agg_config.get('ema_beta', 0.8)
        )

        logger.info(f"[FAP-v3] Module initialized: w1={fair_agg_config.get('w1', 1.0)}, "
                   f"w2={fair_agg_config.get('w2', 0.3)}, w3={fair_agg_config.get('w3', 0.5)}")
    else:
        logger.info("[FAP-v3] Fair aggregation disabled, using standard FedAvg")
    # ================================================================

    # Use the provided experiment directory
    output_dir = exp_dir

    # Initialize global theta using model-provided filter (excludes head/BN/LoRA)
    theta_global = model.state_dict_theta()

    # Track metrics
    metrics_history = {
        'avg_acc': [],
        'worst_acc': [],
        'variance': [],
        'per_domain_acc': {d: [] for d in domains},
    }

    logger.info(f"Starting v2 federated learning training for {total_rounds} rounds")
    logger.info(f"Configuration: participation={clients_participation}")
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

                # Local training with decoupled phases
                theta_state, phi_state, train_acc = trainer.train_client(
                    domain=domain,
                    dataset=client_dataset,
                    batch_size=batch_size,
                    local_steps=local_steps,
                    decoupled_training=decoupled_training,
                    phi_steps_ratio=phi_steps_ratio
                )

                # Collect updates (both flat and per-domain)
                all_theta_updates.append(theta_state)
                all_theta_weights.append(len(client_dataset))
                domain_theta_updates[domain].append(theta_state)
                domain_theta_weights[domain].append(len(client_dataset))
                domain_phi_updates[domain].append(phi_state)
                domain_phi_weights[domain].append(len(client_dataset))

                # logger.info(f"  Client {client_id}: train_acc={train_acc:.2f}%")

        # Phase 3: Domain-Internal Phi Aggregation (LoRA only)
        for domain in domains:
            if domain_phi_updates[domain]:
                aggregated_phi = aggregate_phi_domain(
                    client_phi_list=domain_phi_updates[domain],
                    client_weights=domain_phi_weights[domain],
                    domain=domain
                )
                edge_manager.set_phi(domain, aggregated_phi)
                logger.info(f"Aggregated phi for domain {domain}")

        # Phase 4: Global θ Aggregation (FedAvg)
        if all_theta_updates:
            theta_global = aggregate_theta(
                client_theta_list=all_theta_updates,
                client_weights=all_theta_weights
            )
            
            logger.info(f"Global theta aggregated (round {round_num})")

        # Phase 5: Evaluation
        domain_accuracies = {}
        domain_features_cache = {}  # Cache for FAP v3 S1 statistics

        for domain in domains:
            # Load global theta
            model.load_state_dict(theta_global, strict=False)

            # Load domain's aggregated phi
            phi_state = edge_manager.get_phi(domain)
            if phi_state is not None:
                model.load_state_dict(phi_state, strict=False)

            # Evaluate on domain's validation set
            # Request features if FAP v3 is enabled for S1 statistic computation
            if use_fair_aggregation and fair_drift_calc is not None:
                eval_result = trainer.evaluate_domain(
                    domain=domain,
                    dataset=val_data[domain],
                    batch_size=batch_size,
                    edge_manager=edge_manager,
                    return_features=True
                )
                val_loss, val_acc, L_e, feats, labels = eval_result
                domain_features_cache[domain] = (feats, labels)
            else:
                val_loss, val_acc, L_e = trainer.evaluate_domain(
                    domain=domain,
                    dataset=val_data[domain],
                    batch_size=batch_size,
                    edge_manager=edge_manager
                )

            domain_accuracies[domain] = val_acc
            logger.info(f"Domain {domain}: val_acc={val_acc:.2f}%, L_e={L_e:.4f}")

            # Update fair aggregation state manager with EMA loss (if enabled)
            if use_fair_aggregation and fair_state_mgr is not None:
                fair_state_mgr.update_loss(domain, val_loss)

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

        # ========== FAP v3 Fair Aggregation State Update (Optional) ==========
        # This section computes advanced fair aggregation metrics using S1
        # statistic aggregation for drift computation.
        if use_fair_aggregation and fair_state_mgr is not None and fair_drift_calc is not None and fair_cloud_agg is not None:
            # Update domain sample counts
            for domain in domains:
                domain_samples = sum(domain_theta_weights.get(domain, [0]))
                fair_state_mgr.set_sample_count(domain, domain_samples)

            # ===== S1 Statistic Aggregation for Drift Computation (A3) =====
            # Compute drift using DriftCalculator with cached features
            for domain in domains:
                if domain in domain_features_cache:
                    feats, labels = domain_features_cache[domain]
                    # Compute S1 statistics (per-class projected feature sums)
                    client_stats = fair_drift_calc.compute_client_statistics(
                        features=feats.to(config.get('system', {}).get('device', 'cuda')),
                        labels=labels.to(config.get('system', {}).get('device', 'cuda'))
                    )
                    # Aggregate to domain level (simulating DC aggregation)
                    fair_drift_calc.aggregate_domain_statistics(domain, [client_stats])
                    # Compute prototypes
                    fair_drift_calc.compute_prototypes(domain)
                    # Update snapshot if needed (every tau_d rounds)
                    fair_drift_calc.update_snapshot(domain)
                    # Compute drift score
                    drift_score = fair_drift_calc.compute_drift(domain)
                    # Reset accumulators for next round
                    fair_drift_calc.reset_domain_accumulators(domain)
                else:
                    drift_score = 0.0

                # Update state manager with computed drift
                fair_state_mgr.set_drift(domain, drift_score)

            # Step drift calculator round counter
            fair_drift_calc.step_round()

            # Get all domain states for cloud aggregation
            fair_states = fair_state_mgr.get_all_states()

            # Compute fair aggregation weights (B2-B5)
            fair_weights = fair_cloud_agg.compute_aggregation_weights(
                L_map=fair_states['L_map'],
                drift_map=fair_states['drift_map'],
                H_map=fair_states['H_map'],
                n_map=fair_states['n_map']
            )

            # Update coverage gaps for next round
            fair_state_mgr.update_coverage_gap(fair_weights)

            # Log detailed fair aggregation metrics
            logger.info(f"[FAP-v3] Round {round_num} fair weights computed:")
            for domain in domains:
                L_e = fair_states['L_map'].get(domain, 0.0)
                drift_e = fair_states['drift_map'].get(domain, 0.0)
                H_e = fair_states['H_map'].get(domain, 0.0)
                alpha_e = fair_weights.get(domain, 1.0 / len(domains))
                logger.info(f"  {domain}: L={L_e:.4f}, Δ={drift_e:.4f}, "
                           f"H={H_e:.4f}, α={alpha_e:.4f}")

        # =====================================================================

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
