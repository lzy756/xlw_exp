"""Main federated learning training loop."""

import os
import random
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
from data.domainnet import DomainNetDataset
from data.partition import build_domain_clients
from core.trainer import LocalTrainer, dc_train_on_offload_pool
from core.aggregator import aggregate_theta, aggregate_phi_domain, aggregate_theta_weighted
from core.edge_manager import EdgeManager
from core.selector import FAPFloatSoftmax
from utils.metrics import per_domain_metrics
from utils.common import set_seed, build_logger, get_git_commit_hash
from utils.transforms import build_transforms


class ExperimentEnv:
    """Lightweight facade for data access and sample counting.
    
    Provides convenience methods for calibration and fair-weighted aggregation
    without modifying existing EdgeManager or data structures.
    """

    def __init__(self, train_data: Dict, config: Dict, domains: List[str]):
        """Initialize experiment environment.
        
        Args:
            train_data: Dictionary of training data per domain
            config: Configuration dictionary
            domains: List of domain names
        """
        self.train_data = train_data
        self.config = config
        self.domains = domains

    def dc_unload_dataset(self, domain: str) -> Optional[DomainNetDataset]:
        """Get DC offload pool dataset for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            DomainNetDataset for DC offload pool, or None if not available
        """
        if domain not in self.train_data:
            return None
        
        dc_pool = self.train_data[domain].get('dc_unload_pool', [])
        if not dc_pool:
            return None
        
        dataset = DomainNetDataset(
            root=self.config['data']['root'],
            indices=dc_pool,
            train=True
        )
        
        return dataset if len(dataset) > 0 else None

    def count_domain_samples_this_round(
        self, 
        domain: str, 
        participating: Dict[str, List[int]]
    ) -> int:
        """Count total samples (UE + DC) used in training this round.
        
        Args:
            domain: Domain name
            participating: Dict mapping domain -> list of participating client IDs
            
        Returns:
            Total sample count for the domain
        """
        if domain not in self.train_data:
            return 0
        
        # Count UE samples from participating clients
        ue_samples = 0
        if domain in participating:
            for client_id in participating[domain]:
                if client_id < len(self.train_data[domain]['clients']):
                    client_data = self.train_data[domain]['clients'][client_id]
                    ue_samples += len(client_data.get('local', []))
        
        # Count DC pool samples if offload enabled
        dc_samples = 0
        if self.config.get('data', {}).get('offload_pool_enabled', False):
            dc_pool = self.train_data[domain].get('dc_unload_pool', [])
            dc_samples = len(dc_pool)
        
        return ue_samples + dc_samples

    def build_loader(
        self, 
        dataset: Dataset, 
        batch_size: int, 
        train: bool = True
    ) -> DataLoader:
        """Build DataLoader with appropriate settings.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size
            train: Whether this is for training (affects shuffling)
            
        Returns:
            DataLoader instance
        """
        num_workers = self.config.get('system', {}).get('num_workers', 4)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=True
        )


def _vectorize_theta(model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Vectorize theta parameters (exclude heads and lora_blocks).
    
    Args:
        model_state: Model state dictionary
        
    Returns:
        Flattened parameter vector
    """
    theta_params = []
    for key, param in sorted(model_state.items()):
        if 'heads.' not in key and 'lora_blocks.' not in key:
            theta_params.append(param.flatten())
    
    return torch.cat(theta_params) if theta_params else torch.tensor([])


def calibrate_on_dc(
    agg_domain: str,
    theta_avg: Dict[str, torch.Tensor],
    cfg: Dict,
    model: nn.Module,
    env: ExperimentEnv,
    logger
) -> Dict[str, torch.Tensor]:
    """Calibrate global backbone on aggregator's DC offload pool.
    
    Post-aggregation fine-tuning with proximal regularization to adapt
    theta to the selected aggregator domain while preventing catastrophic
    forgetting of global knowledge.
    
    Args:
        agg_domain: Selected aggregator domain
        theta_avg: Aggregated global backbone parameters (pre-calibration)
        cfg: Configuration dictionary
        model: Neural network model
        env: ExperimentEnv for data access
        logger: Logger instance
        
    Returns:
        Calibrated theta parameters
    """
    # Extract calibration configuration
    cal_cfg = cfg.get('calibration', {})
    steps = cal_cfg.get('steps', 200)
    batch_size = cal_cfg.get('batch_size', 64)
    mu = cal_cfg.get('mu', 0.01)
    lr = cal_cfg.get('lr')
    if lr is None:
        lr = cfg.get('training', {}).get('lr_theta', 0.0003)
    freeze_phi = cal_cfg.get('freeze_phi', True)
    min_samples = cal_cfg.get('min_samples', 100)
    device = cfg.get('system', {}).get('device', 'cpu')
    
    # Early exit: check if offload pool enabled
    if not cfg.get('data', {}).get('offload_pool_enabled', False):
        logger.info("Calibration skipped: offload pool disabled")
        return theta_avg
    
    # Get DC dataset for aggregator domain
    dc_dataset = env.dc_unload_dataset(agg_domain)
    
    # Early exit: check minimum sample threshold
    if dc_dataset is None or len(dc_dataset) < min_samples:
        logger.info(
            f"Calibration skipped: insufficient samples in {agg_domain} DC pool "
            f"(have {len(dc_dataset) if dc_dataset else 0}, need {min_samples})"
        )
        return theta_avg
    
    logger.info(
        f"Starting calibration on {agg_domain} DC pool "
        f"({len(dc_dataset)} samples, {steps} steps)"
    )
    
    # Load theta_avg into model
    model.load_state_dict(theta_avg, strict=False)
    model = model.to(device)
    model.train()
    
    # Freeze phi parameters (heads and lora_blocks)
    if freeze_phi:
        for name, param in model.named_parameters():
            if 'heads.' in name or 'lora_blocks.' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    # Collect theta parameters for optimizer
    theta_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and 'heads.' not in name and 'lora_blocks.' not in name
    ]
    
    # Create optimizer for theta only
    optimizer = optim.AdamW(theta_params, lr=lr, weight_decay=cfg.get('training', {}).get('weight_decay', 0.0001))
    
    # Vectorize initial theta for proximal term (extract from loaded model, not theta_avg)
    # This includes all model buffers and parameters (e.g., BatchNorm running stats)
    theta_init_state = {
        k: v.clone() for k, v in model.state_dict().items()
        if 'heads.' not in k and 'lora_blocks.' not in k
    }
    theta_init_vec = _vectorize_theta(theta_init_state).to(device)
    
    # Create DataLoader
    dataloader = env.build_loader(dc_dataset, batch_size, train=True)
    data_iter = iter(dataloader)
    
    # Calibration loop
    for step in range(steps):
        # Get batch (restart iterator if exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # DomainNetDataset returns (image, label, domain)
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        
        # Validate labels
        num_classes = cfg['data']['num_classes']
        assert labels.min() >= 0 and labels.max() < num_classes, \
            f"Invalid labels: range [{labels.min()}, {labels.max()}], expected [0, {num_classes})"
        
        # Forward pass using aggregator domain head
        logits = model(images, agg_domain)
        
        # Compute CE loss
        ce_loss = nn.functional.cross_entropy(logits, labels)
        
        # Compute proximal loss: μ/2 * ||θ_cur - θ_avg||²
        current_state = {
            k: v for k, v in model.state_dict().items()
            if 'heads.' not in k and 'lora_blocks.' not in k
        }
        theta_cur_vec = _vectorize_theta(current_state).to(device)
        proximal_loss = (mu / 2.0) * torch.sum((theta_cur_vec - theta_init_vec) ** 2)
        
        # Total loss
        total_loss = ce_loss + proximal_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(theta_params, max_norm=5.0)
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            logger.info(
                f"  Calibration step {step+1}/{steps}: "
                f"ce_loss={ce_loss.item():.4f}, prox_loss={proximal_loss.item():.6f}"
            )
    
    # Extract calibrated theta
    calibrated_theta = {
        k: v.cpu().clone()
        for k, v in model.state_dict().items()
        if 'heads.' not in k and 'lora_blocks.' not in k
    }
    
    logger.info(f"Calibration completed for {agg_domain}")
    
    return calibrated_theta


def run_training(
    config: Dict,
    model: torch.nn.Module,
    train_data: Dict[str, Dict],
    val_data: Dict[str, DomainNetDataset],
    edge_manager: EdgeManager,
    selector: FAPFloatSoftmax,
    trainer: LocalTrainer,
    logger,
    exp_dir: str
) -> Dict:
    """Run federated learning training loop.

    Args:
        config: Configuration dictionary
        model: Neural network model
        train_data: Training data per domain and client
        val_data: Validation data per domain
        edge_manager: EdgeManager instance
        selector: FAPFloatSoftmax selector instance
        trainer: LocalTrainer instance
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
    
    # Use the provided experiment directory
    output_dir = exp_dir

    # Initialize ExperimentEnv for data access and sample counting
    env = ExperimentEnv(train_data, config, config['data']['domains'])

    # Initialize global theta (backbone parameters)
    theta_global = {
        k: v.cpu().clone()
        for k, v in model.state_dict().items()
        if 'heads.' not in k and 'lora_blocks.' not in k
    }

    # Track metrics
    metrics_history = {
        'avg_acc': [],
        'worst_acc': [],
        'variance': [],
        'per_domain_acc': {d: [] for d in config['data']['domains']},
        'selected_aggregators': [],
        'alpha_weights': [],  # Fair-weighted aggregation weights
        'fairness_factors': []  # Fairness factors (q)
    }

    logger.info(f"Starting federated learning training for {total_rounds} rounds")
    logger.info(f"Configuration: K={K}, participation={clients_participation}")
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
        for domain in config['data']['domains']:
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
        domain_theta_updates = {d: [] for d in config['data']['domains']}
        domain_theta_weights = {d: [] for d in config['data']['domains']}
        domain_phi_updates = {d: [] for d in config['data']['domains']}
        domain_phi_weights = {d: [] for d in config['data']['domains']}

        for domain in config['data']['domains']:
            # Load domain's current phi
            phi_state = edge_manager.get_phi(domain)
            if phi_state is not None:
                model.load_state_dict(phi_state, strict=False)

            # Load global theta
            model.load_state_dict(theta_global, strict=False)

            for client_id in participating_clients[domain]:
                # Get client's dataset
                client_data = train_data[domain]['clients'][client_id]
                client_dataset = DomainNetDataset(
                    root=config['data']['root'],
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

                logger.info(f"  Client {client_id}: train_acc={train_acc:.2f}%")

        # Phase 3: Domain-Internal Phi Aggregation
        for domain in config['data']['domains']:
            if domain_phi_updates[domain]:
                aggregated_phi = aggregate_phi_domain(
                    client_phi_list=domain_phi_updates[domain],
                    client_weights=domain_phi_weights[domain],
                    domain=domain
                )
                edge_manager.set_phi(domain, aggregated_phi)
                logger.info(f"Aggregated phi for domain {domain}")

        # Phase 4: Evaluation
        domain_accuracies = {}
        for domain in config['data']['domains']:
            # Load domain's aggregated phi
            phi_state = edge_manager.get_phi(domain)
            if phi_state is not None:
                model.load_state_dict(phi_state, strict=False)

            # Load global theta
            model.load_state_dict(theta_global, strict=False)

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
        for domain in config['data']['domains']:
            metrics_history['per_domain_acc'][domain].append(domain_accuracies[domain])

        # Phase 5: Aggregation Point Selection (every K rounds)
        if round_num % K == 0 and all_theta_updates:
            logger.info(f"Round {round_num}: Aggregator selection")

            # Get metrics for selection
            selection_metrics = edge_manager.get_metrics_for_selection()

            # Log drift metrics per domain
            logger.info(f"Domain drift scores (Δ_e): {selection_metrics['drift_map']}")
            logger.info(f"Domain EMA losses (L_e): {selection_metrics['L_map']}")

            # Select aggregator
            selected_domain, probabilities, scores = selector.select(
                L_map=selection_metrics['L_map'],
                drift_map=selection_metrics['drift_map'],
                cover_map=selection_metrics['cover_map'],
                stay_map=selection_metrics['stay_map']
            )

            logger.info(f"Selected aggregator: {selected_domain}")
            logger.info(f"Selection probabilities: {dict(zip(config['data']['domains'], probabilities))}")
            logger.info(f"Raw scores: {dict(zip(config['data']['domains'], scores))}")

            # Build score_map and n_map for aggregation
            score_map = {d: s for d, s in zip(config['data']['domains'], scores)}
            n_map = {
                d: env.count_domain_samples_this_round(d, participating_clients)
                for d in config['data']['domains']
            }

            # Global theta aggregation (fair-weighted or standard)
            fair_weighting_enabled = config.get('fair_weighting', {}).get('enable', False)
            
            if fair_weighting_enabled:
                logger.info("Using fair-weighted aggregation")
                
                # First, aggregate theta within each domain
                domain_theta_aggregated = []
                active_domains = []
                
                for domain in config['data']['domains']:
                    if domain_theta_updates[domain]:
                        # Aggregate theta for this domain
                        domain_theta = aggregate_theta(
                            client_theta_list=domain_theta_updates[domain],
                            client_weights=domain_theta_weights[domain]
                        )
                        domain_theta_aggregated.append(domain_theta)
                        active_domains.append(domain)
                
                # Then apply fair-weighted aggregation across domains
                if domain_theta_aggregated:
                    # Filter maps to only include active domains
                    active_n_map = {d: n_map[d] for d in active_domains}
                    active_score_map = {d: score_map[d] for d in active_domains}
                    
                    theta_global, alpha, q = aggregate_theta_weighted(
                        theta_e_list=domain_theta_aggregated,
                        domains=active_domains,
                        n_map=active_n_map,
                        score_map=active_score_map,
                        agg_domain=selected_domain,
                        cfg=config
                    )
                    
                    logger.info(f"Aggregation weights (α): {dict(zip(active_domains, alpha))}")
                    logger.info(f"Fairness factors (q): {dict(zip(active_domains, q))}")
                    
                    # Pad with None for inactive domains if needed
                    full_alpha = [None] * len(config['data']['domains'])
                    full_q = [None] * len(config['data']['domains'])
                    for i, domain in enumerate(config['data']['domains']):
                        if domain in active_domains:
                            idx = active_domains.index(domain)
                            full_alpha[i] = alpha[idx]
                            full_q[i] = q[idx]
                    
                    metrics_history['alpha_weights'].append(full_alpha)
                    metrics_history['fairness_factors'].append(full_q)
                else:
                    # No updates to aggregate
                    metrics_history['alpha_weights'].append(None)
                    metrics_history['fairness_factors'].append(None)
            else:
                # Standard FedAvg aggregation
                theta_global = aggregate_theta(
                    client_theta_list=all_theta_updates,
                    client_weights=all_theta_weights
                )
                
                metrics_history['alpha_weights'].append(None)
                metrics_history['fairness_factors'].append(None)

            # Calibration step (if enabled)
            calibration_enabled = config.get('calibration', {}).get('enable', False)
            
            if calibration_enabled:
                logger.info(f"Running calibration on {selected_domain} DC pool")
                theta_global = calibrate_on_dc(
                    agg_domain=selected_domain,
                    theta_avg=theta_global,
                    cfg=config,
                    model=model,
                    env=env,
                    logger=logger
                )

            # Update edge manager
            edge_manager.end_round_with_aggregator(selected_domain)
            metrics_history['selected_aggregators'].append(selected_domain)

            logger.info(f"Global theta aggregated at {selected_domain}")

        # Phase 6: Optional DC Offload Pool Training
        if config.get('data', {}).get('offload_pool_enabled', False):
            for domain in config['data']['domains']:
                dc_pool = train_data[domain].get('dc_unload_pool', [])
                if dc_pool:
                    dc_dataset = DomainNetDataset(
                        root=config['data']['root'],
                        indices=dc_pool,
                        train=True
                    )

                    # Load current phi
                    phi_state = edge_manager.get_phi(domain)
                    if phi_state is not None:
                        model.load_state_dict(phi_state, strict=False)

                    # DC training
                    updated_phi = dc_train_on_offload_pool(
                        model=model,
                        domain=domain,
                        dataset=dc_dataset,
                        batch_size=batch_size,
                        device=config['system']['device']
                    )

                    edge_manager.set_phi(domain, updated_phi)
                    logger.info(f"DC training completed for {domain} offload pool")

        # Phase 7: Checkpointing
        if round_num % checkpoint_interval == 0:
            # Save global theta
            theta_path = os.path.join(
                output_dir, 'checkpoints', f'theta_global_r{round_num}.pt'
            )
            torch.save({'state_dict': theta_global, 'round': round_num}, theta_path)
            logger.info(f"Saved theta checkpoint: {theta_path}")

            # Save domain phi states
            for domain in config['data']['domains']:
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
    for domain in config['data']['domains']:
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