"""Main entry point for baseline FL experiments (FedAvg/FedProx)."""

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
from data.domainnet import DomainNetDataset
from data.partition import build_domain_clients
from baseline.core.loop_baseline import run_baseline_training
from utils.common import set_seed, build_logger
from utils.experiment import ExperimentLogger


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict):
    """Create baseline model.

    Args:
        config: Configuration dictionary

    Returns:
        ResNet18Single or ResNet50Single model instance
    """
    backbone = config['model'].get('backbone', 'resnet18_single')
    num_classes = config['data']['num_classes']
    pretrained = config['model']['pretrained']

    if backbone == 'resnet50_single':
        return ResNet50Single(
            num_classes=num_classes,
            pretrained=pretrained
        )
    else:  # default to resnet18_single
        return ResNet18Single(
            num_classes=num_classes,
            pretrained=pretrained
        )


def prepare_data(config: Dict) -> tuple:
    """Prepare training and validation data.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_data, val_data)
    """
    # Load dataset index
    index_path = os.path.join(config['data']['root'], 'index.json')

    # Create dummy index if it doesn't exist (for testing)
    if not os.path.exists(index_path):
        print(f"Warning: index.json not found at {index_path}")
        print("Creating dummy index for testing...")
        dummy_dataset = DomainNetDataset(config['data']['root'])

    with open(index_path, 'r') as f:
        index = json.load(f)

    # Prepare data for each domain
    train_data = {}
    val_data = {}

    for domain_idx, domain in enumerate(config['data']['domains']):
        # Build client partitions for this domain
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

        # Create validation dataset for the domain
        # Aggregate all validation indices from clients
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
    """Main baseline experiment runner."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='FL-DomainNet Baseline Experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/fedavg_fixed.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--algo',
        type=str,
        choices=['fedavg', 'fedprox'],
        help='Override algorithm (fedavg or fedprox)'
    )
    parser.add_argument(
        '--mu',
        type=float,
        help='Override proximal term coefficient (FedProx only)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        help='Override Dirichlet alpha for data partitioning'
    )
    parser.add_argument(
        '--rounds',
        type=int,
        help='Override number of rounds'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Override device'
    )
    parser.add_argument(
        '--fixed-domain',
        type=str,
        dest='fixed_domain',
        help='Override fixed aggregator domain'
    )
    parser.add_argument(
        '--exp-tag',
        type=str,
        dest='exp_tag',
        help='Experiment tag/name to append to timestamp'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        dest='output_dir',
        help='Override output directory'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Keep a copy of original config for saving
    original_config = copy.deepcopy(config)

    # Apply overrides
    if args.algo:
        config['algo'] = args.algo
        print(f"Using algorithm: {args.algo}")
    if args.mu is not None:
        if 'prox' not in config:
            config['prox'] = {}
        config['prox']['mu'] = args.mu
        print(f"Using μ={args.mu}")
    if args.alpha is not None:
        config['partition']['alpha'] = args.alpha
        print(f"Using Dirichlet α={args.alpha}")
    if args.rounds:
        config['training']['total_rounds'] = args.rounds
        print(f"Training for {args.rounds} rounds")
    if args.device:
        config['system']['device'] = args.device
    if args.fixed_domain:
        if 'selector' not in config:
            config['selector'] = {}
        config['selector']['fixed_domain'] = args.fixed_domain
        print(f"Fixed aggregator: {args.fixed_domain}")
    if args.exp_tag:
        config['logging']['exp_name'] = args.exp_tag

    # Create experiment logger and directory
    exp_logger = ExperimentLogger(config, args)
    exp_dir = exp_logger.create_experiment_dir()
    print(f"Experiment directory: {exp_dir}")

    # Save configurations
    exp_logger.save_configs(original_config, config)

    # Save initial experiment info
    exp_logger.save_experiment_info(status='running')

    # Set random seed
    set_seed(config['system']['seed'])

    # Setup logging
    log_file = os.path.join(exp_dir, 'train.log')
    logger = build_logger('Baseline-FL', log_file)

    algo = config.get('algo', 'fedavg')
    logger.info(f"Starting baseline FL experiment: {algo.upper()}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Domains: {config['data']['domains']}")
    logger.info(f"Device: {config['system']['device']}")

    # Check CUDA availability
    if config['system']['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config['system']['device'] = 'cpu'

    try:
        # Prepare data
        logger.info("Preparing data...")
        train_data, val_data = prepare_data(config)
        logger.info("Data preparation complete")

        # Initialize model
        logger.info("Initializing single-head ResNet18 model...")
        model = create_model(config)

        # Run baseline training
        logger.info("Starting baseline training...")
        metrics = run_baseline_training(
            config=config,
            model=model,
            train_data=train_data,
            val_data=val_data,
            logger=logger,
            exp_dir=exp_dir
        )

        # Print summary
        logger.info("=" * 50)
        logger.info("Training Summary")
        logger.info("=" * 50)
        logger.info(f"Algorithm: {algo.upper()}")
        logger.info(f"Final average accuracy: {metrics['avg_acc'][-1]:.2f}%")
        logger.info(f"Final worst accuracy: {metrics['worst_acc'][-1]:.2f}%")
        logger.info(f"Final variance: {metrics['variance'][-1]:.4f}")

        # Check improvement
        if len(metrics['worst_acc']) > 1:
            worst_improvement = metrics['worst_acc'][-1] - metrics['worst_acc'][0]
            logger.info(f"Worst-domain improvement: {worst_improvement:.2f} pp")

            if metrics['variance'][0] > 0:
                variance_reduction = (metrics['variance'][0] - metrics['variance'][-1]) / metrics['variance'][0] * 100
                logger.info(f"Variance reduction: {variance_reduction:.2f}%")

        logger.info("Experiment completed successfully!")

        # Update experiment info with success status
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
