"""ResNet18 with EAPH-LoRA implementation."""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LoRAConv2d(nn.Module):
    """LoRA adapter for Conv2d layers.

    Implements low-rank adaptation for convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int = 16,
        alpha: float = 16.0,
        stride: int = 1,
        padding: int = 0
    ):
        """Initialize LoRA Conv2d adapter.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            rank: LoRA rank (r)
            alpha: LoRA scaling factor
            stride: Convolution stride
            padding: Convolution padding
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Down projection: in_channels -> rank
        self.lora_down = nn.Conv2d(
            in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False
        )

        # Up projection: rank -> out_channels
        # Use same kernel_size, stride, padding as original conv
        self.lora_up = nn.Conv2d(
            rank, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            LoRA output (to be added as residual to main path)
        """
        # Down -> Up projection with scaling
        return self.lora_up(self.lora_down(x)) * self.scaling


class ResNet18_EAPH(nn.Module):
    """ResNet18 with EAPH-LoRA for domain-specific personalization.

    Attaches LoRA modules to layer4 and creates domain-specific heads.
    """

    def __init__(
        self,
        num_classes: int = 126,
        domains: List[str] = None,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        pretrained: bool = True
    ):
        """Initialize ResNet18 with EAPH-LoRA.

        Args:
            num_classes: Number of output classes
            domains: List of domain names
            lora_rank: LoRA rank
            lora_alpha: LoRA scaling factor
            pretrained: Whether to load pretrained weights
        """
        super().__init__()

        if domains is None:
            domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

        self.num_classes = num_classes
        self.domains = domains
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # Extract layers (freeze pretrained backbone initially)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Feature dimension from ResNet18
        self.feature_dim = 512

        # Attach LoRA to layer4
        self.lora_blocks = nn.ModuleDict()
        self._attach_lora_to_layer4()

        # Create domain-specific classification heads
        self.heads = nn.ModuleDict({
            domain: nn.Linear(self.feature_dim, num_classes)
            for domain in domains
        })

        # Initialize heads
        for head in self.heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def _attach_lora_to_layer4(self):
        """Attach LoRA adapters to layer4 convolutions."""
        # ResNet18 layer4 has 2 BasicBlocks, each with 2 conv layers
        for block_idx, block in enumerate(self.layer4):
            # Attach to conv1 of each block
            conv1 = block.conv1
            self.lora_blocks[f'layer4_{block_idx}_conv1'] = LoRAConv2d(
                in_channels=conv1.in_channels,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size[0],
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                stride=conv1.stride[0],
                padding=conv1.padding[0]
            )

            # Attach to conv2 of each block
            conv2 = block.conv2
            self.lora_blocks[f'layer4_{block_idx}_conv2'] = LoRAConv2d(
                in_channels=conv2.in_channels,
                out_channels=conv2.out_channels,
                kernel_size=conv2.kernel_size[0],
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                stride=conv2.stride[0],
                padding=conv2.padding[0]
            )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head.

        Args:
            x: Input tensor

        Returns:
            512-dimensional feature vector
        """
        # Standard ResNet forward until avgpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Layer4 with LoRA adaptation
        for block_idx, block in enumerate(self.layer4):
            identity = x

            # First conv with LoRA
            out = block.conv1(x)
            if f'layer4_{block_idx}_conv1' in self.lora_blocks:
                out = out + self.lora_blocks[f'layer4_{block_idx}_conv1'](x)
            out = block.bn1(out)
            out = block.relu(out)

            # Second conv with LoRA
            conv2_input = out
            out = block.conv2(out)
            if f'layer4_{block_idx}_conv2' in self.lora_blocks:
                out = out + self.lora_blocks[f'layer4_{block_idx}_conv2'](conv2_input)
            out = block.bn2(out)

            # Handle downsample if present
            if block.downsample is not None:
                identity = block.downsample(x)

            out += identity
            out = block.relu(out)
            x = out

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        """Forward pass with domain-specific head.

        Args:
            x: Input tensor
            domain: Domain name for selecting appropriate head

        Returns:
            Logits for the specified domain
        """
        features = self.forward_features(x)
        return self.heads[domain](features)

    def parameters_theta(self) -> List[nn.Parameter]:
        """Get backbone parameters (θ).

        Returns:
            List of parameters excluding heads and LoRA modules
        """
        params = []
        for name, param in self.named_parameters():
            if 'heads.' not in name and 'lora_blocks.' not in name:
                params.append(param)
        return params

    def parameters_phi(self, domain: str) -> List[nn.Parameter]:
        """Get domain-specific parameters (φ_e).

        Args:
            domain: Domain name

        Returns:
            List of LoRA and domain head parameters
        """
        params = []

        # Add LoRA parameters
        for param in self.lora_blocks.parameters():
            params.append(param)

        # Add domain-specific head parameters
        if domain in self.heads:
            for param in self.heads[domain].parameters():
                params.append(param)

        return params