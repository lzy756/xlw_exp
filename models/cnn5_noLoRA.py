"""5-layer CNN WITHOUT LoRA adapters (ablation study).

A simple 5-layer CNN model that uses standard linear classifier instead of LoRA.
This serves as an ablation baseline to demonstrate the effectiveness of LoRA adapters.

Key difference from cnn5_domainheads.py:
- Classifier uses standard nn.Linear instead of LoRALinear
- parameters_phi() returns empty list (no domain-specific params)
- All parameters are globally aggregated (standard FedAvg)
"""

from typing import Dict, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN5_NoLoRA(nn.Module):
    """5-layer CNN with standard classifier (no LoRA) for ablation study.

    Architecture:
        Conv1: 3 -> 32 channels, 3x3 kernel, stride 1, padding 1
        Conv2: 32 -> 64 channels, 3x3 kernel, stride 1, padding 1
        Conv3: 64 -> 128 channels, 3x3 kernel, stride 1, padding 1
        Conv4: 128 -> 256 channels, 3x3 kernel, stride 1, padding 1
        Conv5: 256 -> 512 channels, 3x3 kernel, stride 1, padding 1
        Global Average Pooling
        Dropout (p=0.5)
        Standard Linear Classifier (no LoRA)

    Each conv layer is followed by BatchNorm, ReLU, and MaxPool2d.
    """

    def __init__(
        self,
        num_classes: int = 7,
        domains: List[str] = None,
        pretrained: bool = False,  # Ignored, kept for API compatibility
        adapter_rank: int = 8,  # Ignored, kept for API compatibility
        in_channels: int = 3
    ):
        """Initialize 5-layer CNN without LoRA.

        Args:
            num_classes: Number of output classes
            domains: List of domain names (kept for API compatibility)
            pretrained: Ignored, kept for API compatibility
            adapter_rank: Ignored, kept for API compatibility
            in_channels: Number of input channels (default 3 for RGB)
        """
        super().__init__()

        if domains is None:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']

        self.num_classes = num_classes
        self.domains = domains
        self.adapter_rank = adapter_rank  # Kept for compatibility

        # Convolutional layers (Backbone)
        self.conv1 = self._make_conv_block(in_channels, 32)
        self.conv2 = self._make_conv_block(32, 64)
        self.conv3 = self._make_conv_block(64, 128)
        self.conv4 = self._make_conv_block(128, 256)
        self.conv5 = self._make_conv_block(256, 512)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension
        self._feature_dim = 512

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)

        # Standard Linear Classifier (NO LoRA)
        self.classifier = nn.Linear(self._feature_dim, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create a convolutional block with BN, ReLU, and MaxPool.

        Args:
            in_ch: Input channels
            out_ch: Output channels

        Returns:
            Sequential block: Conv -> BN -> ReLU -> MaxPool
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def _initialize_weights(self) -> None:
        """Initialize backbone weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (512 for this CNN)."""
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim features.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Feature tensor of shape (N, 512)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor, domain: str = None) -> torch.Tensor:
        """Forward pass (domain ignored since no LoRA).

        Args:
            x: Input tensor of shape (N, C, H, W)
            domain: Ignored, kept for API compatibility

        Returns:
            Logits tensor of shape (N, num_classes)
        """
        # 1. Extract features
        features = self.forward_features(x)

        # 2. Apply Dropout
        features = self.dropout(features)

        # 3. Standard classifier (no domain-specific adaptation)
        return self.classifier(features)

    def parameters_theta(self) -> List[nn.Parameter]:
        """Get globally aggregated parameters (θ).

        For no-LoRA model, ALL parameters are globally aggregated.
        """
        return list(self.parameters())

    def parameters_bn(self) -> List[nn.Parameter]:
        """Get BatchNorm parameters (kept local per FedBN).

        Returns:
            List of BatchNorm parameters
        """
        params = []
        for name, param in self.named_parameters():
            # BatchNorm parameters in Sequential blocks (index 1)
            if '.1.weight' in name or '.1.bias' in name:
                params.append(param)
        return params

    def parameters_phi(self, domain: str) -> List[nn.Parameter]:
        """Get domain-specific parameters (φ_e).

        For no-LoRA model, returns empty list since there are no
        domain-specific parameters.
        """
        return []

    def state_dict_theta(self) -> Dict[str, torch.Tensor]:
        """Export globally aggregated state dict.

        For no-LoRA model, returns ALL parameters since everything
        is globally aggregated.
        """
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
        }

    def state_dict_bn(self) -> Dict[str, torch.Tensor]:
        """Export BatchNorm state dict (kept local per FedBN).

        Returns:
            State dict containing BatchNorm parameters and buffers
        """
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if '.1.weight' in k or '.1.bias' in k
            or '.1.running' in k or '.1.num_batches' in k
        }

    def state_dict_phi(self, domain: str) -> Dict[str, torch.Tensor]:
        """Export domain-specific state dict.

        For no-LoRA model, returns empty dict since there are no
        domain-specific parameters.
        """
        return {}
