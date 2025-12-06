"""5-layer CNN with domain-specific LoRA adapters.

A simple 5-layer CNN model without pretrained weights, with support for
domain-specific Low-Rank Adaptation (LoRA) on the classifier layer.

This implements standard LoRA: W_eff = W_0 + B @ A
- W_0 (base weight) is globally aggregated
- B, A (low-rank matrices) are domain-specific and kept local
- B is initialized to zero so the model starts equivalent to baseline
"""

from typing import Dict, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation.

    Implements: output = (W_0 + B @ A) @ x + bias
    where W_0 is the base weight and B @ A is the low-rank adaptation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        domains: List[str] = None
    ):
        """Initialize LoRA linear layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: Rank for low-rank adaptation
            domains: List of domain names
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.domains = domains if domains else ['default']

        # Base weight (W_0) - globally shared and aggregated
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # LoRA matrices (A, B) - domain-specific, kept local
        # A: (rank, in_features), B: (out_features, rank)
        self.lora_A = nn.ParameterDict({
            d: nn.Parameter(torch.Tensor(rank, in_features))
            for d in self.domains
        })
        self.lora_B = nn.ParameterDict({
            d: nn.Parameter(torch.Tensor(out_features, rank))
            for d in self.domains
        })

        # Scaling factor for stability
        self.scaling = 1.0 / rank

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        # Base weight: Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA initialization (critical!)
        for d in self.domains:
            # A: Kaiming initialization
            nn.init.kaiming_uniform_(self.lora_A[d], a=math.sqrt(5))
            # B: Zero initialization (ensures model starts as baseline)
            nn.init.zeros_(self.lora_B[d])

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor of shape (N, in_features)
            domain: Domain name for LoRA selection

        Returns:
            Output tensor of shape (N, out_features)
        """
        # Base path: W_0 @ x + b
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA path: (B @ A) @ x * scaling
        if domain in self.lora_A:
            A = self.lora_A[domain]  # (rank, in_features)
            B = self.lora_B[domain]  # (out_features, rank)

            # Efficient computation: (x @ A.T) @ B.T
            lora_out = (x @ A.t()) @ B.t()

            return base_out + lora_out * self.scaling
        else:
            return base_out


class CNN5_DomainHeads(nn.Module):
    """5-layer CNN with LoRA-adapted classifier for domain personalization.

    Architecture:
        Conv1: 3 -> 32 channels, 3x3 kernel, stride 1, padding 1
        Conv2: 32 -> 64 channels, 3x3 kernel, stride 1, padding 1
        Conv3: 64 -> 128 channels, 3x3 kernel, stride 1, padding 1
        Conv4: 128 -> 256 channels, 3x3 kernel, stride 1, padding 1
        Conv5: 256 -> 512 channels, 3x3 kernel, stride 1, padding 1
        Global Average Pooling
        Dropout (p=0.5)
        LoRA Classifier: W_eff = W_0 + B @ A

    Each conv layer is followed by BatchNorm, ReLU, and MaxPool2d.
    LoRA is applied only to the classifier layer for domain personalization.
    """

    def __init__(
        self,
        num_classes: int = 7,
        domains: List[str] = None,
        pretrained: bool = False,  # Ignored, kept for API compatibility
        adapter_rank: int = 8,
        in_channels: int = 3
    ):
        """Initialize 5-layer CNN with LoRA classifier.

        Args:
            num_classes: Number of output classes
            domains: List of domain names
            pretrained: Ignored, kept for API compatibility
            adapter_rank: Rank for LoRA adaptation
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

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)

        # LoRA Classifier (replaces fc_global + adapters)
        self.classifier = LoRALinear(
            in_features=self._feature_dim,
            out_features=num_classes,
            rank=adapter_rank,
            domains=domains
        )

        # Initialize backbone weights
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
        # Note: LoRALinear handles its own initialization

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
        """Forward pass with domain-specific LoRA adaptation.

        Args:
            x: Input tensor of shape (N, C, H, W)
            domain: Domain name for LoRA selection

        Returns:
            Logits tensor of shape (N, num_classes)
        """
        # 1. Extract features
        features = self.forward_features(x)

        # 2. Apply Dropout
        features = self.dropout(features)

        # 3. LoRA classifier: (W_0 + B @ A) @ features + bias
        return self.classifier(features, domain)

    def parameters_theta(self) -> List[nn.Parameter]:
        """Get globally aggregated parameters (θ) with FedBN strategy.

        Returns backbone conv weights + base classifier weights (W_0, bias).
        Excludes:
        - LoRA matrices (A, B) - domain-specific
        - BatchNorm parameters - local statistics (FedBN)

        Returns:
            List of parameters for global aggregation
        """
        params = []
        for name, param in self.named_parameters():
            # Exclude LoRA matrices (domain-specific)
            if 'lora_A' in name or 'lora_B' in name:
                continue
            # Exclude classifier base weights/bias (kept local per FedRep head)
            if name.startswith('classifier.weight') or name.startswith('classifier.bias'):
                continue
            # Exclude BatchNorm parameters (FedBN strategy)
            # In Sequential blocks, index 1 is BatchNorm2d
            if '.1.weight' in name or '.1.bias' in name or '.1.running' in name:
                continue
            params.append(param)
        return params

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

        Returns LoRA matrices (A, B) and classifier base weights/bias
        for the specified domain (FedRep-style head local loop).

        Args:
            domain: Domain name

        Returns:
            List of LoRA parameters for the domain
        """
        return [
            self.classifier.weight,
            self.classifier.bias,
            self.classifier.lora_A[domain],
            self.classifier.lora_B[domain]
        ]

    def state_dict_theta(self) -> Dict[str, torch.Tensor]:
        """Export globally aggregated state dict with FedBN.

        Returns:
            State dict containing backbone conv + base classifier weights.
            Excludes LoRA and BatchNorm parameters.
        """
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if 'lora_A' not in k and 'lora_B' not in k
            and not k.startswith('classifier.weight')
            and not k.startswith('classifier.bias')
            and '.1.weight' not in k and '.1.bias' not in k
            and '.1.running' not in k and '.1.num_batches' not in k
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

        Args:
            domain: Domain name

        Returns:
            State dict containing LoRA matrices for the specified domain
        """
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith('classifier.weight') or k.startswith('classifier.bias')
            or f'lora_A.{domain}' in k or f'lora_B.{domain}' in k
        }
