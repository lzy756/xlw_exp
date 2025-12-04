"""ResNet50/ResNet18 with shared global head + lightweight domain bias.

Shared backbone + single global classifier (θ), plus per-domain bias (φ_e).
This keeps most capacity global while allowing small domain-specific
adjustment without fragmenting data across full heads.
"""

from typing import Dict, List
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ResNet50_DomainHeads(nn.Module):
    """ResNet50 with global head + per-domain low-rank adapters."""

    def __init__(
        self,
        num_classes: int = 126,
        domains: List[str] = None,
        pretrained: bool = True,
        adapter_rank: int = 4
    ):
        """Initialize ResNet50 with domain-specific adapters.

        Args:
            num_classes: Number of output classes
            domains: List of domain names
            pretrained: Whether to load pretrained weights
            adapter_rank: Rank for per-domain low-rank adapters
        """
        super().__init__()
        
        if domains is None:
            domains = ['clipart', 'infograph', 'painting', 
                      'quickdraw', 'real', 'sketch']
        
        self.num_classes = num_classes
        self.domains = domains
        self.adapter_rank = adapter_rank
        
        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Backbone layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self._feature_dim = 2048
        
        # Shared global classifier
        self.fc_global = nn.Linear(self._feature_dim, num_classes)
        nn.init.xavier_uniform_(self.fc_global.weight)
        nn.init.zeros_(self.fc_global.bias)

        # Lightweight per-domain bias (personalization)
        self.biases = nn.ParameterDict({
            domain: nn.Parameter(torch.zeros(num_classes))
            for domain in domains
        })

        # Per-domain low-rank adapters: A_d (C x r) and B_d (r x C)
        self.adapters_down = nn.ModuleDict({
            domain: nn.Linear(self._feature_dim, adapter_rank, bias=False)
            for domain in domains
        })
        self.adapters_up = nn.ModuleDict({
            domain: nn.Linear(adapter_rank, num_classes, bias=False)
            for domain in domains
        })

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (2048 for ResNet50)."""
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dim features.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Feature tensor of shape (N, 2048)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        """Forward pass with domain-specific head.

        Args:
            x: Input tensor of shape (N, C, H, W)
            domain: Domain name for head selection

        Returns:
            Logits tensor of shape (N, num_classes)
        """
        features = self.forward_features(x)
        # Low-rank adapter for domain
        adapter = self.adapters_up[domain](self.adapters_down[domain](features))
        return self.fc_global(features) + adapter + self.biases[domain]

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


class ResNet18_DomainHeads(nn.Module):
    """ResNet18 with global head + per-domain low-rank adapters."""

    def __init__(
        self,
        num_classes: int = 126,
        domains: List[str] = None,
        pretrained: bool = True,
        adapter_rank: int = 4
    ):
        """Initialize ResNet18 with domain-specific heads.

        Args:
            num_classes: Number of output classes
            domains: List of domain names
            pretrained: Whether to load pretrained weights
            adapter_rank: Rank for per-domain low-rank adapters
        """
        super().__init__()
        
        if domains is None:
            domains = ['clipart', 'infograph', 'painting', 
                      'quickdraw', 'real', 'sketch']
        
        self.num_classes = num_classes
        self.domains = domains
        self.adapter_rank = adapter_rank
        
        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Backbone layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        self._feature_dim = 512
        
        # Shared global classifier
        self.fc_global = nn.Linear(self._feature_dim, num_classes)
        nn.init.xavier_uniform_(self.fc_global.weight)
        nn.init.zeros_(self.fc_global.bias)

        # Lightweight per-domain bias
        self.biases = nn.ParameterDict({
            domain: nn.Parameter(torch.zeros(num_classes))
            for domain in domains
        })

        # Per-domain low-rank adapters
        self.adapters_down = nn.ModuleDict({
            domain: nn.Linear(self._feature_dim, adapter_rank, bias=False)
            for domain in domains
        })
        self.adapters_up = nn.ModuleDict({
            domain: nn.Linear(adapter_rank, num_classes, bias=False)
            for domain in domains
        })

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (512 for ResNet18)."""
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim features.

        Args:
            x: Input tensor of shape (N, C, H, W)

        Returns:
            Feature tensor of shape (N, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        """Forward pass with domain-specific head.

        Args:
            x: Input tensor of shape (N, C, H, W)
            domain: Domain name for head selection

        Returns:
            Logits tensor of shape (N, num_classes)
        """
        features = self.forward_features(x)
        adapter = self.adapters_up[domain](self.adapters_down[domain](features))
        return self.fc_global(features) + adapter + self.biases[domain]

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


class MobileNetV3_DomainHeads(nn.Module):
    """MobileNetV3-Small with global head + per-domain low-rank adapters."""

    def __init__(
        self,
        num_classes: int = 126,
        domains: List[str] = None,
        pretrained: bool = True,
        adapter_rank: int = 4
    ):
        super().__init__()
        if domains is None:
            domains = ['clipart', 'infograph', 'painting',
                       'quickdraw', 'real', 'sketch']
        self.num_classes = num_classes
        self.domains = domains
        self.adapter_rank = adapter_rank

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = mobilenet_v3_small(weights=weights)

        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._feature_dim = mobilenet.classifier[0].in_features  # 576

        # Shared global classifier
        self.fc_global = nn.Linear(self._feature_dim, num_classes)
        nn.init.xavier_uniform_(self.fc_global.weight)
        nn.init.zeros_(self.fc_global.bias)

        # Domain-specific adapters + bias
        self.biases = nn.ParameterDict({
            domain: nn.Parameter(torch.zeros(num_classes))
            for domain in domains
        })
        self.adapters_down = nn.ModuleDict({
            domain: nn.Linear(self._feature_dim, adapter_rank, bias=False)
            for domain in domains
        })
        self.adapters_up = nn.ModuleDict({
            domain: nn.Linear(adapter_rank, num_classes, bias=False)
            for domain in domains
        })

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        feats = self.forward_features(x)
        adapter = self.adapters_up[domain](self.adapters_down[domain](feats))
        return self.fc_global(feats) + adapter + self.biases[domain]

    def parameters_theta(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters()
                if not n.startswith('biases.') and not n.startswith('adapters_')]

    def parameters_phi(self, domain: str) -> List[nn.Parameter]:
        return [
            self.biases[domain],
            *self.adapters_down[domain].parameters(),
            *self.adapters_up[domain].parameters()
        ]

    def state_dict_theta(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if not k.startswith('biases.') and not k.startswith('adapters_')
        }

    def state_dict_phi(self, domain: str) -> Dict[str, torch.Tensor]:
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if k == f'biases.{domain}'
            or k.startswith(f'adapters_down.{domain}')
            or k.startswith(f'adapters_up.{domain}')
        }
