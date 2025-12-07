"""5-layer CNN for FedAvg/FedProx baseline experiments.

A simple 5-layer CNN model without pretrained weights to demonstrate
federated learning performance under domain shift and distribution
heterogeneity. Unlike pretrained ResNet models, this model starts from
random initialization and is more sensitive to these challenges.
"""

from typing import Dict, List
import torch
import torch.nn as nn


class CNN5Single(nn.Module):
    """5-layer CNN with single global head for federated learning.

    Architecture:
        Conv1: 3 -> 32 channels, 3x3 kernel, stride 1, padding 1
        Conv2: 32 -> 64 channels, 3x3 kernel, stride 1, padding 1
        Conv3: 64 -> 128 channels, 3x3 kernel, stride 1, padding 1
        Conv4: 128 -> 256 channels, 3x3 kernel, stride 1, padding 1
        Conv5: 256 -> 512 channels, 3x3 kernel, stride 1, padding 1
        Global Average Pooling
        FC: 512 -> num_classes

    Each conv layer is followed by BatchNorm, ReLU, and MaxPool2d.
    This results in feature maps of size H/32 x W/32 before GAP.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = False,  # Kept for API compatibility, ignored
        in_channels: int = 3
    ):
        """Initialize 5-layer CNN with single global head.

        Args:
            num_classes: Number of output classes
            pretrained: Ignored, kept for API compatibility
            in_channels: Number of input channels (default 3 for RGB)
        """
        super().__init__()

        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = self._make_conv_block(in_channels, 32)
        self.conv2 = self._make_conv_block(32, 64)
        self.conv3 = self._make_conv_block(64, 128)
        self.conv4 = self._make_conv_block(128, 256)
        self.conv5 = self._make_conv_block(256, 512)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension
        self._feature_dim = 512

        # Single global classification head
        self.fc = nn.Linear(self._feature_dim, num_classes)

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

    @property
    def feature_dim(self) -> int:
        """Get feature dimension for compatibility.

        Returns:
            Feature dimension (512 for this CNN)
        """
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head.

        Args:
            x: Input tensor of shape (N, 3, H, W)

        Returns:
            512-dimensional feature vector of shape (N, 512)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x: torch.Tensor, domain: str | None = None) -> torch.Tensor:
        """Forward pass with global head.

        Args:
            x: Input tensor of shape (N, 3, H, W)
            domain: Domain name (ignored, kept for API compatibility)

        Returns:
            Logits of shape (N, num_classes)
        """
        features = self.forward_features(x)
        return self.fc(features)

    def state_dict_global(self) -> Dict[str, torch.Tensor]:
        """Export all model parameters.

        Returns:
            State dictionary with all parameters
        """
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def load_state_dict_global(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load all model parameters.

        Args:
            state_dict: State dictionary to load
        """
        self.load_state_dict(state_dict, strict=True)

    def parameters_all(self) -> List[nn.Parameter]:
        """Get all trainable parameters.

        Returns:
            List of all model parameters
        """
        return list(self.parameters())

    def state_dict_fedsak_shared(self, shared_part: str = 'fc') -> Dict[str, torch.Tensor]:
        """Export shared layers for FedSAK aggregation.

        Args:
            shared_part: Which part to share
                - 'fc': Only final classifier (DH scenario, efficiency)
                - 'all': All parameters (DH scenario, full model)
                - 'backbone': Only conv layers (TH scenario)

        Returns:
            State dict of shared parameters
        """
        full_state = self.state_dict()

        if shared_part == 'fc':
            # DH scenario (efficient): only share FC layer
            return {
                k: v.cpu().clone()
                for k, v in full_state.items()
                if k.startswith('fc.')
            }
        elif shared_part == 'all':
            # DH scenario (full): share entire model
            return {k: v.cpu().clone() for k, v in full_state.items()}
        elif shared_part == 'backbone':
            # TH scenario: share conv layers only
            return {
                k: v.cpu().clone()
                for k, v in full_state.items()
                if k.startswith('conv') or k.startswith('avgpool')
            }
        else:
            raise ValueError(f"Unknown shared_part: {shared_part}")

    def load_state_dict_fedsak_shared(
        self,
        state_dict: Dict[str, torch.Tensor],
        shared_part: str = 'fc'
    ) -> None:
        """Load FedSAK-regularized shared layers.

        Args:
            state_dict: Regularized state dict from server
            shared_part: Which part was shared (must match export)
        """
        self.load_state_dict(state_dict, strict=False)
