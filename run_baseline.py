"""Entry point for baseline FedAvg/FedProx experiments.

Runs single-head federated learning with optional proximal term and
fixed aggregation point selection. Uses the same data partitioning
pipeline as the main experiment for fair comparison.
"""

import os
import sys
import json
import copy
import argparse
from typing import Dict

import torch
import yaml

from baseline.models.resnet18_single import ResNet18Single
from baseline.models.resnet50_single import ResNet50Single
from baseline.core.loop_baseline import run_baseline_training
from data.domainnet import DomainNetDataset
from data.partition import build_domain_clients
from utils.common import set_seed, build_logger
from utils.experiment import ExperimentLogger


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: Dict):
    """Create baseline single-head model."""
    backbone = config['model'].get('backbone', 'resnet18')
    num_classes = config['data']['num_classes']
    pretrained = config['model'].get('pretrained', True)

    if backbone in ('resnet18', 'resnet18_single'):
        return ResNet18Single(num_classes=num_classes, pretrained=pretrained)
    elif backbone in ('resnet50', 'resnet50_single'):
        return ResNet50Single(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


def prepare_data(config: Dict):
    """Prepare train/val splits per domain using existing partition logic."""
    index_path = os.path.join(config['data']['root'], 'index.json')
    if not os.path.exists(index_path):
        # Build dummy index to keep flow consistent in testing
        _ = DomainNetDataset(config['data']['root'])

    with open(index_path, 'r') as f:
        index = json.load(f)

    train_data = {}
    val_data = {}

    for domain_idx, domain in enumerate(config['data']['domains']):
        domain_data = build_domain_clients(
            index=index,
            domain=domain,
            num_clients=config['partition']['num_clients_per_domain'],
            alpha=config['partition']['alpha'],
            unload_ratio=config['partition']['unload_ratio'],
            val_ratio=config['partition']['val_ratio'],
            seed=config['partition']['seed'] + domain_idx
        )
        train_data[domain] = domain_data

        val_indices = []
        for client_data in domain_data['clients'].values():
            val_indices.extend(client_data['val'])

        val_data[domain] = DomainNetDataset(
            root=config['data']['root'],
            indices=val_indices,
            train=False
        )

    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description="Baseline FL (FedAvg/FedProx)")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--domains', type=str, nargs='+', help='Override domains')
    parser.add_argument('--rounds', type=int, help='Override total rounds')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Override device')
    parser.add_argument('--exp-tag', type=str, dest='exp_tag', help='Experiment tag/name')
    parser.add_argument('--output-dir', type=str, dest='output_dir', help='Override output directory')
    parser.add_argument('--no-timestamp', action='store_true', dest='no_timestamp', help='Disable timestamp dir')
    parser.add_argument('--algo', type=str, choices=['fedavg', 'fedprox'], help='Override algorithm')
    parser.add_argument('--mu', type=float, help='Override FedProx mu')
    parser.add_argument('--fixed-domain', type=str, dest='fixed_domain', help='Override fixed aggregator domain')
    parser.add_argument('--lr', type=float, help='Override learning rate for baseline')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    original_config = copy.deepcopy(config)

    # Apply overrides
    if args.domains:
        config['data']['domains'] = args.domains
        print(f"Using domains: {args.domains}")
    if args.rounds:
        config['training']['total_rounds'] = args.rounds
        print(f"Training for {args.rounds} rounds")
    if args.device:
        config['system']['device'] = args.device
    if args.algo:
        config['algo'] = args.algo
    if args.mu is not None:
        config.setdefault('prox', {})['mu'] = args.mu
    if args.fixed_domain:
        config.setdefault('selector', {})['fixed_domain'] = args.fixed_domain
    if args.lr is not None:
        config.setdefault('training', {})['lr'] = args.lr

    # Experiment logger / directory
    exp_logger = ExperimentLogger(config, args)
    exp_dir = exp_logger.create_experiment_dir()
    print(f"Experiment directory: {exp_dir}")
    exp_logger.save_configs(original_config, config)
    exp_logger.save_experiment_info(status='running')

    # Seed and logger
    set_seed(config['system']['seed'])
    log_file = os.path.join(exp_dir, 'train.log')
    logger = build_logger('Baseline-FL', log_file)

    logger.info("Starting Baseline FL experiment")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Domains: {config['data']['domains']}")
    logger.info(f"Device: {config['system']['device']}")
    logger.info(f"Algorithm: {config.get('algo', 'fedavg')}")

    # Device check
    if config['system']['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config['system']['device'] = 'cpu'
    elif config['system']['device'] == 'cuda':
        torch.backends.cudnn.benchmark = True

    try:
        logger.info("Preparing data...")
        train_data, val_data = prepare_data(config)
        logger.info("Data preparation complete")

        logger.info("Initializing baseline model...")
        model = create_model(config)

        metrics = run_baseline_training(
            config=config,
            model=model,
            train_data=train_data,
            val_data=val_data,
            logger=logger,
            exp_dir=exp_dir
        )

        # Save metrics
        metrics_path = os.path.join(exp_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # Summary
        logger.info("="*50)
        logger.info("Training Summary")
        logger.info("="*50)
        logger.info(f"Final average accuracy: {metrics['avg_acc'][-1]:.2f}%")
        logger.info(f"Final worst accuracy: {metrics['worst_acc'][-1]:.2f}%")
        logger.info(f"Final variance: {metrics['variance'][-1]:.4f}")

        exp_logger.save_experiment_info(status='completed', exit_code=0)
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        exp_logger.save_experiment_info(status='interrupted', exit_code=130)
        raise
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        exp_logger.save_experiment_info(status='failed', error=str(e), exit_code=1)
        raise


if __name__ == '__main__':
    main()
