"""Main entry point for FL-DomainNet experiment."""

import os
import sys
import json
import argparse
from typing import Dict
import torch
import yaml

from models.resnet18_lora import ResNet18_EAPH
from data.domainnet import DomainNetDataset
from data.partition import build_domain_clients
from core.trainer import LocalTrainer
from core.edge_manager import EdgeManager
from core.selector import FAPFloatSoftmax
from core.loop import run_training
from utils.common import set_seed, build_logger


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
    """Main experiment runner."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='FL-DomainNet Experiment')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--domains',
        type=str,
        nargs='+',
        help='Override domains to use (e.g., --domains clipart real)'
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
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply overrides
    if args.domains:
        config['data']['domains'] = args.domains
        print(f"Using domains: {args.domains}")
    if args.rounds:
        config['training']['total_rounds'] = args.rounds
        print(f"Training for {args.rounds} rounds")
    if args.device:
        config['system']['device'] = args.device

    # Set random seed
    set_seed(config['system']['seed'])

    # Setup logging
    log_dir = os.path.join(
        config['logging']['output_dir'],
        config['logging']['exp_name']
    )
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logger = build_logger('FL-DomainNet', log_file)

    logger.info("Starting FL-DomainNet experiment")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Domains: {config['data']['domains']}")
    logger.info(f"Device: {config['system']['device']}")

    # Check CUDA availability
    if config['system']['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config['system']['device'] = 'cpu'

    # Prepare data
    logger.info("Preparing data...")
    train_data, val_data = prepare_data(config)
    logger.info(f"Data preparation complete")

    # Initialize model
    logger.info("Initializing model...")
    model = ResNet18_EAPH(
        num_classes=config['data']['num_classes'],
        domains=config['data']['domains'],
        lora_rank=config['model']['lora']['rank'],
        lora_alpha=config['model']['lora']['alpha'],
        pretrained=config['model']['pretrained']
    )

    # Initialize components
    trainer = LocalTrainer(
        model=model,
        device=config['system']['device'],
        lr_theta=config['training']['lr_theta'],
        lr_phi=config['training']['lr_phi'],
        weight_decay=config['training']['weight_decay'],
        cosine_lr=config['training']['cosine_lr']
    )

    edge_manager = EdgeManager(
        model=model,
        domains=config['data']['domains'],
        num_classes=config['data']['num_classes'],
        proj_dim=config['edge_manager']['proj_dim'],
        device=config['system']['device']
    )

    selector = FAPFloatSoftmax(
        domains=config['data']['domains'],
        w1=config['selector']['w1'],
        w2=config['selector']['w2'],
        w3=config['selector']['w3'],
        w4=config['selector']['w4'],
        tau=config['selector']['tau']
    )

    # Run training
    logger.info("Starting training...")
    metrics = run_training(
        config=config,
        model=model,
        train_data=train_data,
        val_data=val_data,
        edge_manager=edge_manager,
        selector=selector,
        trainer=trainer,
        logger=logger
    )

    # Save final metrics
    metrics_path = os.path.join(log_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Training Summary")
    logger.info("="*50)
    logger.info(f"Final average accuracy: {metrics['avg_acc'][-1]:.2f}%")
    logger.info(f"Final worst accuracy: {metrics['worst_acc'][-1]:.2f}%")
    logger.info(f"Final variance: {metrics['variance'][-1]:.4f}")
    logger.info(f"Selected aggregators: {metrics['selected_aggregators']}")

    # Check success criteria
    if len(metrics['worst_acc']) > 1:
        worst_improvement = metrics['worst_acc'][-1] - metrics['worst_acc'][0]
        variance_reduction = (metrics['variance'][0] - metrics['variance'][-1]) / metrics['variance'][0] * 100

        logger.info(f"\nWorst-domain improvement: {worst_improvement:.2f} pp")
        logger.info(f"Variance reduction: {variance_reduction:.2f}%")

    logger.info("\nExperiment completed successfully!")


if __name__ == '__main__':
    main()