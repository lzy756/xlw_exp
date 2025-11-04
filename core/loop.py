"""Main federated learning training loop."""

import os
import random
from typing import Dict, List, Optional
import torch
import yaml
from data.domainnet import DomainNetDataset
from data.partition import build_domain_clients
from core.trainer import LocalTrainer, dc_train_on_offload_pool
from core.aggregator import aggregate_theta, aggregate_phi_domain
from core.edge_manager import EdgeManager
from core.selector import FAPFloatSoftmax
from utils.metrics import per_domain_metrics
from utils.common import set_seed, build_logger, get_git_commit_hash


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
        'selected_aggregators': []
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

                # Collect updates
                all_theta_updates.append(theta_state)
                all_theta_weights.append(len(client_dataset))
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

            # Global theta aggregation
            theta_global = aggregate_theta(
                client_theta_list=all_theta_updates,
                client_weights=all_theta_weights
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