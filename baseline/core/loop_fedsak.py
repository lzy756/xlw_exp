"""FedSAK baseline training loop.

Implements the FedSAK algorithm with tensor trace norm regularization
for federated learning under data/model/task heterogeneity.
"""

import os
import random
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.factory import create_dataset
from core.aggregator_fedsak import aggregate_fedsak
from utils.metrics import per_domain_metrics
from utils.common import get_git_commit_hash


class LocalTrainerFedSAK:
    """Local trainer for FedSAK (standard SGD, no decoupled training)."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        lr: float = 0.01,
        weight_decay: float = 1e-4
    ):
        """Initialize FedSAK local trainer.

        Args:
            model: Neural network model
            device: Device for training
            lr: Learning rate
            weight_decay: Weight decay
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()

    def train_client(
        self,
        dataset,
        batch_size: int = 32,
        local_steps: int = 3
    ):
        """Train on local data with standard SGD.

        Args:
            dataset: Local training dataset
            batch_size: Batch size
            local_steps: Number of local epochs

        Returns:
            Tuple of (train_accuracy, data_size)
        """
        self.model.train()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=0.9
        )

        correct = 0
        total = 0

        for step in range(local_steps):
            for images, labels, _ in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        train_acc = 100.0 * correct / total if total > 0 else 0.0
        return train_acc, len(dataset)

    def evaluate(self, dataset, batch_size: int = 32):
        """Evaluate on validation data.

        Args:
            dataset: Validation dataset
            batch_size: Batch size

        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        self.model.eval()

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device == 'cuda' else False
        )

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        val_loss = total_loss / total if total > 0 else 0.0
        val_acc = 100.0 * correct / total if total > 0 else 0.0

        return val_loss, val_acc


def run_fedsak_training(
    config: Dict,
    model: nn.Module,
    train_data: Dict,
    val_data: Dict,
    logger,
    exp_dir: str
) -> Dict:
    """Run FedSAK baseline training.

    Args:
        config: Configuration dictionary
        model: Neural network model
        train_data: Training data per domain
        val_data: Validation data per domain
        logger: Logger instance
        exp_dir: Experiment directory

    Returns:
        Dictionary with training metrics
    """
    # Extract config
    total_rounds = config['training']['total_rounds']
    local_steps = config['training']['local_steps']
    batch_size = config['training']['batch_size']
    clients_participation = config['training']['clients_participation']
    checkpoint_interval = config['logging']['checkpoint_interval']
    domains = config['data']['domains']
    device = config['system']['device']

    # FedSAK specific params
    fedsak_config = config.get('fedsak', {})
    lr_w = fedsak_config.get('lr_w', 0.01)
    lam = fedsak_config.get('lambda', 6.0)
    shared_part = fedsak_config.get('shared_part', 'fc')
    lr_local = config['training'].get('lr', 0.01)

    # Initialize trainer
    trainer = LocalTrainerFedSAK(
        model=model,
        device=device,
        lr=lr_local,
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )

    # Track per-client regularized states
    client_states_registry = {}

    # Metrics
    metrics_history = {
        'avg_acc': [],
        'worst_acc': [],
        'variance': [],
        'per_domain_acc': {d: [] for d in domains}
    }

    logger.info(f"Starting FedSAK training for {total_rounds} rounds")
    logger.info(f"FedSAK config: lr_w={lr_w}, lambda={lam}, shared_part={shared_part}")
    logger.info(f"Git commit: {get_git_commit_hash()}")

    # Initialize ALL clients (not just participating ones)
    # Each client maintains its own personalized model throughout training
    all_client_keys = []
    for domain in domains:
        num_clients = len(train_data[domain]['clients'])
        for client_id in range(num_clients):
            client_key = f"{domain}_{client_id}"
            all_client_keys.append(client_key)
            # Initialize each client's shared layer state (t=0)
            client_states_registry[client_key] = model.state_dict_fedsak_shared(shared_part=shared_part)
    
    logger.info(f"Initialized {len(all_client_keys)} clients with personalized models")

    # Main loop
    for round_num in range(1, total_rounds + 1):
        logger.info('='*50)
        logger.info(f"Round {round_num}/{total_rounds}")
        logger.info('='*50)

        # Phase 1: Client sampling
        participating_clients = {}
        for domain in domains:
            domain_clients = train_data[domain]['clients']
            num_clients = len(domain_clients)
            num_selected = max(1, int(num_clients * clients_participation))
            selected_ids = random.sample(range(num_clients), num_selected)
            participating_clients[domain] = selected_ids
            logger.info(f"Domain {domain}: selected {len(selected_ids)} clients")

        # Phase 2: Client-side loop (接收 -> 替换 -> 训练 -> 上传)
        client_uploads = []  # w_i^t (训练后的共享层)
        client_ids = []
        client_sizes = []

        for domain in domains:
            for client_id in participating_clients[domain]:
                client_key = f"{domain}_{client_id}"

                # Step 1: 接收并替换 (Receive & Reload)
                # 将服务端上一轮处理后的共享层 w_tilde^{t-1} 加载到本地模型
                model.load_state_dict_fedsak_shared(
                    client_states_registry[client_key],
                    shared_part=shared_part
                )

                # Step 2: 本地训练 (Local Training)
                # 对整个模型进行 SGD 训练，适应本地数据分布
                client_data = train_data[domain]['clients'][client_id]
                client_dataset = create_dataset(
                    config=config,
                    indices=client_data['local'],
                    train=True
                )

                if len(client_dataset) == 0:
                    logger.warning(f"Client {client_key} has no data, skipping")
                    continue

                train_acc, data_size = trainer.train_client(
                    dataset=client_dataset,
                    batch_size=batch_size,
                    local_steps=local_steps
                )

                # Step 3: 提取并上传 (Extract & Upload)
                # 上传的是训练后的共享层参数 w_i^t，而不是梯度
                w_i = model.state_dict_fedsak_shared(shared_part=shared_part)
                client_uploads.append(w_i)
                client_ids.append(client_key)
                client_sizes.append(data_size)

        # Phase 3: Server-side loop (堆叠 -> 计算迹范数梯度 -> 正则化更新 -> 分发)
        if client_uploads:
            logger.info(f"Server aggregating {len(client_uploads)} client uploads via FedSAK")
            
            # Server Step 1-3: 堆叠 + 计算梯度 + 更新
            # 注意：这不是加权平均！而是 w_tilde_i = w_i - lr_w * lambda * grad_trace_norm
            updated_states = aggregate_fedsak(
                client_states=client_uploads,
                lr_w=lr_w,
                lam=lam,
                device=device
            )

            # Server Step 4: 分发回客户端
            # 将修正后的 w_tilde_i^t 存储，供下一轮使用
            for client_key, w_tilde_i in zip(client_ids, updated_states):
                client_states_registry[client_key] = w_tilde_i

        # Phase 4: Evaluation (每个客户端用自己的个性化模型测试)
        # FedSAK 是个性化 FL，没有全局模型，每个客户端测试自己的模型
        domain_accuracies = {domain: [] for domain in domains}
        
        for domain in domains:
            domain_clients = train_data[domain]['clients']
            for client_id in range(len(domain_clients)):
                client_key = f"{domain}_{client_id}"
                
                # 加载该客户端的个性化模型
                model.load_state_dict_fedsak_shared(
                    client_states_registry[client_key],
                    shared_part=shared_part
                )

                # 在该客户端的验证集上测试
                _, val_acc = trainer.evaluate(
                    dataset=val_data[domain],
                    batch_size=batch_size
                )
                
                domain_accuracies[domain].append(val_acc)
        
        # 计算每个 domain 的平均准确率
        domain_avg_acc = {}
        for domain in domains:
            if domain_accuracies[domain]:
                avg = sum(domain_accuracies[domain]) / len(domain_accuracies[domain])
                domain_avg_acc[domain] = avg
                logger.info(f"Domain {domain}: val_acc={avg:.2f}% (avg of {len(domain_accuracies[domain])} clients)")

        # Aggregate metrics
        avg_acc, worst_acc, variance = per_domain_metrics(domain_avg_acc)
        logger.info(f"Round {round_num} metrics:")
        logger.info(f"  Average accuracy: {avg_acc:.2f}%")
        logger.info(f"  Worst accuracy: {worst_acc:.2f}%")
        logger.info(f"  Variance: {variance:.4f}")

        metrics_history['avg_acc'].append(avg_acc)
        metrics_history['worst_acc'].append(worst_acc)
        metrics_history['variance'].append(variance)
        for domain in domains:
            metrics_history['per_domain_acc'][domain].append(domain_avg_acc.get(domain, 0.0))

        # Phase 5: Checkpointing
        # 保存每个客户端的个性化模型（可选择性保存代表客户端）
        if round_num % checkpoint_interval == 0:
            checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 为每个 domain 保存第一个客户端的模型作为代表
            for domain in domains:
                client_key = f"{domain}_0"
                if client_key in client_states_registry:
                    state_path = os.path.join(
                        checkpoint_dir, f'fedsak_{domain}_client0_r{round_num}.pt'
                    )
                    torch.save({
                        'state_dict': client_states_registry[client_key],
                        'round': round_num,
                        'client_key': client_key
                    }, state_path)

    logger.info("FedSAK training completed!")
    return metrics_history
