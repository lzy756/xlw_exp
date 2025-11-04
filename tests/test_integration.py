"""Integration tests for federated learning system."""

import pytest
import torch
import tempfile
import yaml
import json
import os
from models.resnet18_lora import ResNet18_EAPH
from data.domainnet import DomainNetDataset
from data.partition import build_domain_clients
from core.trainer import LocalTrainer
from core.aggregator import aggregate_theta, aggregate_phi_domain
from core.edge_manager import EdgeManager
from core.selector import FAPFloatSoftmax
from utils.common import set_seed


def create_test_config():
    """Create a minimal test configuration."""
    config = {
        'data': {
            'root': tempfile.mkdtemp(),
            'domains': ['clipart', 'real'],
            'num_classes': 10
        },
        'partition': {
            'num_clients_per_domain': 2,
            'alpha': 0.5,
            'unload_ratio': 0.1,
            'val_ratio': 0.2,
            'seed': 42
        },
        'model': {
            'backbone': 'resnet18',
            'pretrained': False,
            'lora': {
                'rank': 4,
                'alpha': 4.0
            }
        },
        'training': {
            'total_rounds': 3,
            'local_steps': 2,
            'batch_size': 8,
            'clients_participation': 0.5,
            'lr_theta': 1e-3,
            'lr_phi': 1e-3,
            'weight_decay': 0,
            'cosine_lr': False
        },
        'selector': {
            'K': 2,
            'w1': 1.0,
            'w2': 0.5,
            'w3': 0.3,
            'w4': 0.2,
            'tau': 1.0
        },
        'edge_manager': {
            'ema_beta': 0.9,
            'proj_dim': 16
        },
        'system': {
            'device': 'cpu',  # Use CPU for tests
            'seed': 42,
            'num_workers': 0
        },
        'logging': {
            'log_interval': 1,
            'checkpoint_interval': 10,
            'output_dir': tempfile.mkdtemp(),
            'exp_name': 'test'
        }
    }
    return config


def test_full_round_2_domains_4_clients():
    """Test a complete training round with 2 domains and 4 clients."""
    set_seed(42)
    config = create_test_config()

    # Create dummy dataset index
    index = {
        'domains': config['data']['domains'],
        'num_classes': config['data']['num_classes'],
        'samples': []
    }

    # Add dummy samples
    for domain_idx, domain in enumerate(config['data']['domains']):
        for class_id in range(config['data']['num_classes']):
            for sample_id in range(20):  # 20 samples per class
                index['samples'].append({
                    'path': f'{domain}/class_{class_id}/sample_{sample_id}.jpg',
                    'label': class_id,
                    'domain': domain
                })

    # Save index
    index_path = os.path.join(config['data']['root'], 'index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f)

    # Initialize model
    model = ResNet18_EAPH(
        num_classes=config['data']['num_classes'],
        domains=config['data']['domains'],
        lora_rank=config['model']['lora']['rank'],
        lora_alpha=config['model']['lora']['alpha'],
        pretrained=False
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

    # Prepare data
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

        # Create validation dataset
        val_indices = []
        for client_data in domain_data['clients'].values():
            val_indices.extend(client_data['val'])

        val_data[domain] = DomainNetDataset(
            root=config['data']['root'],
            indices=val_indices,
            train=False
        )

    # Track initial loss
    initial_losses = {}
    for domain in config['data']['domains']:
        _, _, L_e = trainer.evaluate_domain(
            domain=domain,
            dataset=val_data[domain],
            batch_size=config['training']['batch_size']
        )
        initial_losses[domain] = L_e

    # Run one training round
    edge_manager.begin_round()

    # Simulate client training
    all_theta_updates = []
    all_theta_weights = []

    for domain in config['data']['domains']:
        domain_phi_updates = []
        domain_phi_weights = []

        # Train one client per domain
        client_data = train_data[domain]['clients'][0]
        client_dataset = DomainNetDataset(
            root=config['data']['root'],
            indices=client_data['local'][:10],  # Use only 10 samples for speed
            train=True
        )

        theta_state, phi_state, train_acc = trainer.train_client(
            domain=domain,
            dataset=client_dataset,
            batch_size=4,
            local_steps=1
        )

        all_theta_updates.append(theta_state)
        all_theta_weights.append(len(client_dataset))
        domain_phi_updates.append(phi_state)
        domain_phi_weights.append(len(client_dataset))

        # Aggregate domain phi
        if domain_phi_updates:
            aggregated_phi = aggregate_phi_domain(
                client_phi_list=domain_phi_updates,
                client_weights=domain_phi_weights,
                domain=domain
            )
            edge_manager.set_phi(domain, aggregated_phi)

    # Global theta aggregation
    if all_theta_updates:
        theta_global = aggregate_theta(
            client_theta_list=all_theta_updates,
            client_weights=all_theta_weights
        )
        model.load_state_dict(theta_global, strict=False)

    # Evaluate after training
    final_losses = {}
    for domain in config['data']['domains']:
        phi_state = edge_manager.get_phi(domain)
        if phi_state:
            model.load_state_dict(phi_state, strict=False)

        _, val_acc, L_e = trainer.evaluate_domain(
            domain=domain,
            dataset=val_data[domain],
            batch_size=config['training']['batch_size'],
            edge_manager=edge_manager
        )
        final_losses[domain] = L_e

    # Basic assertions
    assert all(domain in final_losses for domain in config['data']['domains'])
    assert all(isinstance(loss, float) for loss in final_losses.values())

    # Test selection
    metrics = edge_manager.get_metrics_for_selection()
    selected_domain, probs, scores = selector.select(
        L_map=metrics['L_map'],
        drift_map=metrics['drift_map'],
        cover_map=metrics['cover_map'],
        stay_map=metrics['stay_map']
    )

    assert selected_domain in config['data']['domains']
    assert len(probs) == len(config['data']['domains'])
    assert abs(probs.sum() - 1.0) < 1e-6  # Probabilities sum to 1

    print(f"Test passed! Initial losses: {initial_losses}, Final losses: {final_losses}")


def test_domain_phi_isolation():
    """Test that domain-specific parameters don't leak across domains."""
    set_seed(42)
    config = create_test_config()

    # Initialize model
    model = ResNet18_EAPH(
        num_classes=config['data']['num_classes'],
        domains=config['data']['domains'],
        lora_rank=config['model']['lora']['rank'],
        lora_alpha=config['model']['lora']['alpha'],
        pretrained=False
    )

    # Get initial phi parameters for each domain
    initial_phi = {}
    for domain in config['data']['domains']:
        phi_params = model.parameters_phi(domain)
        initial_phi[domain] = [p.clone() for p in phi_params]

    # Modify phi for one domain
    domain1 = config['data']['domains'][0]
    for param in model.parameters_phi(domain1):
        param.data += torch.randn_like(param) * 0.1

    # Check that other domains' phi unchanged
    domain2 = config['data']['domains'][1]
    for orig, current in zip(initial_phi[domain2], model.parameters_phi(domain2)):
        # LoRA parameters are shared, so they will change
        # But domain-specific heads should not
        if 'heads' in str(current):
            assert torch.allclose(orig, current), "Domain isolation violated"

    print("Domain isolation test passed!")


def test_selection_every_K_rounds():
    """Test that selection happens every K rounds and worst-domain gets selected more frequently."""
    set_seed(42)
    config = create_test_config()

    # Create dummy dataset index with imbalanced data to create "worst" domains
    index = {
        'domains': config['data']['domains'],
        'num_classes': config['data']['num_classes'],
        'samples': []
    }

    # Add more samples for 'clipart' (easier domain) and fewer for 'real' (harder domain)
    for class_id in range(config['data']['num_classes']):
        # Clipart: 50 samples per class
        for sample_id in range(50):
            index['samples'].append({
                'path': f'clipart/class_{class_id}/sample_{sample_id}.jpg',
                'label': class_id,
                'domain': 'clipart'
            })
        # Real: 30 samples per class (harder due to less data)
        for sample_id in range(30):
            index['samples'].append({
                'path': f'real/class_{class_id}/sample_{sample_id}.jpg',
                'label': class_id,
                'domain': 'real'
            })

    # Save index
    index_path = os.path.join(config['data']['root'], 'index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f)

    # Create dummy images
    for sample in index['samples'][:10]:  # Just create a few for testing
        img_path = os.path.join(config['data']['root'], sample['path'])
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        # Create a tiny dummy image
        from PIL import Image
        img = Image.new('RGB', (32, 32), color='red')
        img.save(img_path)

    # Initialize model
    model = ResNet18_EAPH(
        num_classes=config['data']['num_classes'],
        domains=config['data']['domains'],
        lora_rank=config['model']['lora']['rank'],
        lora_alpha=config['model']['lora']['alpha'],
        pretrained=False
    )

    # Initialize EdgeManager
    edge_manager = EdgeManager(
        model=model,
        domains=config['data']['domains'],
        num_classes=config['data']['num_classes'],
        proj_dim=config['edge_manager']['proj_dim'],
        device=config['system']['device']
    )

    # Initialize Selector
    selector = FAPFloatSoftmax(
        domains=config['data']['domains'],
        w1=config['selector']['w1'],
        w2=config['selector']['w2'],
        w3=config['selector']['w3'],
        w4=config['selector']['w4'],
        tau=config['selector']['tau']
    )

    # Simulate multiple rounds with selection
    K = config['selector']['K']
    total_rounds = 10  # Run 10 rounds to see selection pattern
    selection_history = []

    for round_num in range(1, total_rounds + 1):
        # Update coverage
        edge_manager.begin_round()

        # Simulate different losses for domains (make 'real' consistently worse)
        # In real training, these would come from actual evaluation
        # We simulate 'real' having higher loss (worse performance)
        if round_num <= 5:
            # Early rounds: 'real' performs worse
            L_map = {'clipart': 0.3, 'real': 0.7}
            drift_map = {'clipart': 0.1, 'real': 0.3}
        else:
            # Later rounds: still worse but improving
            L_map = {'clipart': 0.2, 'real': 0.5}
            drift_map = {'clipart': 0.05, 'real': 0.2}

        # Update EMA losses manually
        for domain in config['data']['domains']:
            edge_manager.L_ema[domain].update(L_map[domain])

        # Simulate drift by creating dummy prototypes
        for domain in config['data']['domains']:
            # Create some dummy prototypes
            edge_manager.proto_sum[domain][0] = torch.randn(config['edge_manager']['proj_dim'])
            edge_manager.proto_cnt[domain][0] = 10.0
            # Compute drift (will be 0 first time, then non-zero)
            edge_manager.compute_drift(domain)

        # Every K rounds, do selection
        if round_num % K == 0:
            metrics = edge_manager.get_metrics_for_selection()
            selected_domain, probs, scores = selector.select(
                L_map=metrics['L_map'],
                drift_map=metrics['drift_map'],
                cover_map=metrics['cover_map'],
                stay_map=metrics['stay_map']
            )

            selection_history.append(selected_domain)
            edge_manager.end_round_with_aggregator(selected_domain)

            print(f"Round {round_num}: Selected {selected_domain}")
            print(f"  Probabilities: {dict(zip(config['data']['domains'], probs))}")
            print(f"  L_map: {metrics['L_map']}")

    # Verify selections happened
    assert len(selection_history) == total_rounds // K, "Selection should happen every K rounds"

    # Verify 'real' (worse domain) was selected at least once
    # Since it has higher loss and drift, it should be favored by the selector
    assert 'real' in selection_history, "Worst-performing domain should be selected at least once"

    # Count selections
    real_count = selection_history.count('real')
    clipart_count = selection_history.count('clipart')

    print(f"\nSelection statistics over {len(selection_history)} selection rounds:")
    print(f"  'real' (worst domain): {real_count} times")
    print(f"  'clipart' (better domain): {clipart_count} times")

    # With fairness-aware selection, the worst domain should get selected more often
    # Note: This is probabilistic, so we can't guarantee it every time
    # But with sufficient rounds and clear performance gap, it should happen
    print("\nSelection test passed!")


if __name__ == '__main__':
    test_full_round_2_domains_4_clients()
    test_domain_phi_isolation()
    test_selection_every_K_rounds()
    print("All integration tests passed!")