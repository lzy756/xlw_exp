"""Baseline federated learning training loop.

Implements FedAvg and FedProx with fixed aggregation point,
supporting UE clients + DC clients (offload pool).
"""

import os
import random
from typing import Dict, List, Optional
import torch
import json
from torch.utils.data import Dataset

from data.factory import create_dataset
from baseline.core.trainer_fedavg import LocalTrainerFedAvg
from baseline.core.trainer_fedprox import LocalTrainerFedProx
from baseline.core.selector_fixed import FixedSelector
from core.aggregator import fedavg
from utils.metrics import per_domain_metrics
from utils.common import get_git_commit_hash


def run_baseline_training(
    config: Dict,
    model: torch.nn.Module,
    train_data: Dict[str, Dict],
    val_data: Dict[str, Dataset],
    logger,
    exp_dir: str
) -> Dict:
    """Run baseline federated learning training loop.

    Args:
        config: Configuration dictionary
        model: Neural network model (ResNet18Single)
        train_data: Training data per domain and client
        val_data: Validation data per domain
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
    device = config['system']['device']

    # Algorithm configuration
    algo = config.get('algo', 'fedavg')
    fixed_domain = config.get('selector', {}).get('fixed_domain', 'real')
    mu = config.get('prox', {}).get('mu', 0.01)

    # Create selector
    selector = FixedSelector(fixed_domain=fixed_domain)

    # Create trainer based on algorithm
    lr = config['training'].get('lr', config['training'].get('lr_theta', 3e-4))
    weight_decay = config['training'].get('weight_decay', 1e-4)

    if algo == 'fedprox':
        trainer = LocalTrainerFedProx(
            model=model,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            mu=mu
        )
        logger.info(f"Using FedProx with μ={mu}")
    else:
        trainer = LocalTrainerFedAvg(
            model=model,
            device=device,
            lr=lr,
            weight_decay=weight_decay
        )
        logger.info("Using FedAvg")

    # Initialize global model state
    w_global = model.state_dict_global()

    # Track metrics
    domains = config['data']['domains']
    metrics_history = {
        'avg_acc': [],
        'worst_acc': [],
        'variance': [],
        'per_domain_acc': {d: [] for d in domains},
        'selected_aggregators': [],
    }

    logger.info(f"Starting baseline FL training for {total_rounds} rounds")
    logger.info(f"Algorithm: {algo}, Fixed aggregator: {fixed_domain}")
    logger.info(f"Git commit: {get_git_commit_hash()}")

    # Main training loop
    for round_num in range(1, total_rounds + 1):
        logger.info('='*50)
        logger.info(f"Round {round_num}/{total_rounds}")
        logger.info('='*50)

        # Load global model state for this round
        trainer.load_global_state(w_global)

        # Phase 1: Client Sampling
        participating_clients = {}
        for domain in domains:
            domain_clients = train_data[domain]['clients']
            num_clients = len(domain_clients)
            num_selected = max(1, int(num_clients * clients_participation))

            # Random sampling
            selected_ids = random.sample(range(num_clients), num_selected)
            participating_clients[domain] = selected_ids

            logger.info(f"Domain {domain}: selected {len(selected_ids)} UE clients")

        # Phase 2: Local Training
        all_client_states = []
        all_client_weights = []

        # 2.1 UE Client Training
        for domain in domains:
            for client_id in participating_clients[domain]:
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

                # Load global state before training
                trainer.load_global_state(w_global)

                # Local training
                if algo == 'fedprox':
                    state_dict, num_samples = trainer.train_client(
                        dataset=client_dataset,
                        batch_size=batch_size,
                        local_steps=local_steps,
                        w_global_snapshot=w_global
                    )
                else:
                    state_dict, num_samples = trainer.train_client(
                        dataset=client_dataset,
                        batch_size=batch_size,
                        local_steps=local_steps
                    )

                all_client_states.append(state_dict)
                all_client_weights.append(num_samples)

        # 2.2 DC Client Training (one per domain, always participates)
        if config.get('data', {}).get('offload_pool_enabled', False):
            for domain in domains:
                dc_pool = train_data[domain].get('dc_unload_pool', [])
                if not dc_pool:
                    continue

                dc_dataset = create_dataset(
                    config=config,
                    indices=dc_pool,
                    train=True
                )

                if len(dc_dataset) == 0:
                    continue

                # Load global state before training
                trainer.load_global_state(w_global)

                # DC client training
                if algo == 'fedprox':
                    state_dict, num_samples = trainer.train_client(
                        dataset=dc_dataset,
                        batch_size=batch_size,
                        local_steps=local_steps,
                        w_global_snapshot=w_global
                    )
                else:
                    state_dict, num_samples = trainer.train_client(
                        dataset=dc_dataset,
                        batch_size=batch_size,
                        local_steps=local_steps
                    )

                all_client_states.append(state_dict)
                all_client_weights.append(num_samples)
                logger.info(f"DC client {domain}: {num_samples} samples")

        # Phase 3: Global Aggregation (FedAvg)
        if all_client_states:
            w_global = fedavg(all_client_states, all_client_weights)
            logger.info(f"Aggregated {len(all_client_states)} client updates")

        # Get selected aggregator (always fixed)
        agg_domain = selector.select()
        metrics_history['selected_aggregators'].append(agg_domain)

        # Phase 4: Evaluation
        domain_accuracies = {}
        trainer.load_global_state(w_global)

        for domain in domains:
            val_loss, val_acc = trainer.evaluate(
                dataset=val_data[domain],
                batch_size=batch_size
            )
            domain_accuracies[domain] = val_acc
            logger.info(f"Domain {domain}: val_acc={val_acc:.2f}%, val_loss={val_loss:.4f}")

        # Calculate aggregate metrics
        avg_acc, worst_acc, variance = per_domain_metrics(domain_accuracies)
        logger.info(f"Round {round_num} metrics:")
        logger.info(f"  Average accuracy: {avg_acc:.2f}%")
        logger.info(f"  Worst accuracy: {worst_acc:.2f}%")
        logger.info(f"  Variance: {variance:.4f}")
        logger.info(f"  Aggregator: {agg_domain} (fixed)")

        # Store metrics
        metrics_history['avg_acc'].append(avg_acc)
        metrics_history['worst_acc'].append(worst_acc)
        metrics_history['variance'].append(variance)
        for domain in domains:
            metrics_history['per_domain_acc'][domain].append(domain_accuracies[domain])

        # Phase 5: Checkpointing
        if round_num % checkpoint_interval == 0:
            checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save global model
            checkpoint_path = os.path.join(
                checkpoint_dir, f'w_global_r{round_num}.pt'
            )
            torch.save({
                'state_dict': w_global,
                'round': round_num,
                'algo': algo
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Final checkpoint
    logger.info("Training completed! Saving final checkpoints...")

    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save final global model
    final_path = os.path.join(checkpoint_dir, 'w_global_final.pt')
    torch.save({
        'state_dict': w_global,
        'round': total_rounds,
        'algo': algo,
        'git_commit': get_git_commit_hash()
    }, final_path)

    # Save metrics
    metrics_path = os.path.join(exp_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)

    logger.info(f"All checkpoints saved to {checkpoint_dir}")
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics_history
