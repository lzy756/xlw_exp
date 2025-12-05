"""5-layer CNN with domain-specific heads and adapters.

A simple 5-layer CNN model without pretrained weights, with support for
domain-specific biases and low-rank feature adapters. Unlike pretrained ResNet
models, this model starts from random initialization and is more
sensitive to domain shift and distribution heterogeneity.

The feature adapters map features to a low-rank space and back, adding a
residual to adapt shared features to domain-specific distributions.
"""

from typing import Dict, List
import torch
import torch.nn as nn


class CNN5_DomainHeads(nn.Module):
    """5-layer CNN with global head + per-domain low-rank feature adapters.

    Architecture:
        Conv1: 3 -> 32 channels, 3x3 kernel, stride 1, padding 1
        Conv2: 32 -> 64 channels, 3x3 kernel, stride 1, padding 1
        Conv3: 64 -> 128 channels, 3x3 kernel, stride 1, padding 1
        Conv4: 128 -> 256 channels, 3x3 kernel, stride 1, padding 1
        Conv5: 256 -> 512 channels, 3x3 kernel, stride 1, padding 1
        Global Average Pooling
        Dropout (p=0.5)
        Feature Adapter: F_adapted = F + Adapter(F)
        FC Global: 512 -> num_classes
        Per-domain bias

    Each conv layer is followed by BatchNorm, ReLU, and MaxPool2d.
    Designed for PACS from scratch training.
    """

    def __init__(
        self,
        num_classes: int = 7,
        domains: List[str] = None,
        pretrained: bool = False,  # Ignored, kept for API compatibility
        adapter_rank: int = 16,  # Larger rank for feature-level adaptation
        in_channels: int = 3
    ):
        """Initialize 5-layer CNN with domain-specific feature adapters.

        Args:
            num_classes: Number of output classes
            domains: List of domain names
            pretrained: Ignored, kept for API compatibility
            adapter_rank: Rank for per-domain low-rank feature adapters
            in_channels: Number of input channels (default 3 for RGB)
        """
        super().__init__()

        if domains is None:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']

        self.num_classes = num_classes
        self.domains = domains
        self.adapter_rank = adapter_rank

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

        # Dropout to prevent overfitting on PACS
        self.dropout = nn.Dropout(p=0.5)

        # Shared global classifier
        self.fc_global = nn.Linear(self._feature_dim, num_classes)

        # Lightweight per-domain bias (personalization)
        self.biases = nn.ParameterDict({
            domain: nn.Parameter(torch.zeros(num_classes))
            for domain in domains
        })

        # Per-domain low-rank feature adapters: F_adapted = F + Up(Act(Down(F)))
        # Maps features to low-rank space and back as residual
        self.adapters_down = nn.ModuleDict({
            domain: nn.Linear(self._feature_dim, adapter_rank, bias=False)
            for domain in domains
        })
        self.adapters_up = nn.ModuleDict({
            domain: nn.Linear(adapter_rank, self._feature_dim, bias=False)
            for domain in domains
        })

        # Activation function for adapter
        self.adapter_act = nn.ReLU()

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
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize adapter up-projection to near-zero to preserve shared features at start
        for domain in self.domains:
            nn.init.zeros_(self.adapters_up[domain].weight)

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

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        """Forward pass with domain-specific feature adaptation.

        Args:
            x: Input tensor of shape (N, C, H, W)
            domain: Domain name for adapter selection

        Returns:
            Logits tensor of shape (N, num_classes)
        """
        # 1. Extract shared features
        features = self.forward_features(x)

        # 2. Apply Dropout
        features = self.dropout(features)

        # 3. Apply feature adapter (Residual connection: F_adapted = F + Adapter(F))
        # This aligns shared features to domain-specific distribution
        adapter_h = self.adapters_down[domain](features)
        adapter_h = self.adapter_act(adapter_h)
        delta_features = self.adapters_up[domain](adapter_h)

        adapted_features = features + delta_features

        # 4. Shared classifier + domain bias
        return self.fc_global(adapted_features) + self.biases[domain]

    def parameters_theta(self) -> List[nn.Parameter]:
        """Get backbone parameters (θ) for global aggregation.

        Returns:
            List of backbone parameters (excludes heads)
        """
        params = []
        for name, param in self.named_parameters():
            if not name.startswith('biases.') and not name.startswith('adapters_'):
                params.append(param)
        return params

    def parameters_phi(self, domain: str) -> List[nn.Parameter]:
        """Get domain-specific head parameters (φ_e).

        Args:
            domain: Domain name

        Returns:
            List of parameters for the domain's head
        """
        return [
            self.biases[domain],
            *self.adapters_down[domain].parameters(),
            *self.adapters_up[domain].parameters()
        ]

    def state_dict_theta(self) -> Dict[str, torch.Tensor]:
        """Export backbone state dict.

        Returns:
            State dict containing only backbone parameters
        """
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if not k.startswith('biases.') and not k.startswith('adapters_')
        }

    def state_dict_phi(self, domain: str) -> Dict[str, torch.Tensor]:
        """Export domain head state dict.

        Args:
            domain: Domain name

        Returns:
            State dict containing only the specified domain's head parameters
        """
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if k == f'biases.{domain}'
            or k.startswith(f'adapters_down.{domain}')
            or k.startswith(f'adapters_up.{domain}')
        }
