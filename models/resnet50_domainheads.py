"""ResNet50 with domain-specific heads (no LoRA).

This module implements the v2 simplified architecture:
- Backbone θ: ResNet50 conv1 → layer4 → avgpool (2048-dim features)
- Heads φ_e: One Linear(2048, num_classes) per domain

This follows the FedPer paradigm: shared backbone + personalized heads.
"""

from typing import Dict, List
import torch
import torch.nn as nn
from torchvision import models


class ResNet50_DomainHeads(nn.Module):
    """ResNet50 with domain-specific classification heads.
    
    Architecture:
    - Backbone θ: ResNet50 conv1 → layer4 → avgpool (2048-dim features)
    - Heads φ_e: One Linear(2048, num_classes) per domain
    
    This follows the FedPer paradigm: shared backbone + personalized heads.
    """

    def __init__(
        self,
        num_classes: int = 126,
        domains: List[str] = None,
        pretrained: bool = True
    ):
        """Initialize ResNet50 with domain-specific heads.

        Args:
            num_classes: Number of output classes
            domains: List of domain names
            pretrained: Whether to load pretrained weights
        """
        super().__init__()
        
        if domains is None:
            domains = ['clipart', 'infograph', 'painting', 
                      'quickdraw', 'real', 'sketch']
        
        self.num_classes = num_classes
        self.domains = domains
        
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
        
        # Domain-specific heads
        self.heads = nn.ModuleDict({
            domain: nn.Linear(self._feature_dim, num_classes)
            for domain in domains
        })
        
        # Initialize heads
        for head in self.heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

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
        return self.heads[domain](features)

    def parameters_theta(self) -> List[nn.Parameter]:
        """Get backbone parameters (θ) for global aggregation.

        Returns:
            List of backbone parameters (excludes heads)
        """
        params = []
        for name, param in self.named_parameters():
            if 'heads.' not in name:
                params.append(param)
        return params

    def parameters_phi(self, domain: str) -> List[nn.Parameter]:
        """Get domain-specific head parameters (φ_e).

        Args:
            domain: Domain name

        Returns:
            List of parameters for the domain's head
        """
        return list(self.heads[domain].parameters())

    def state_dict_theta(self) -> Dict[str, torch.Tensor]:
        """Export backbone state dict.

        Returns:
            State dict containing only backbone parameters
        """
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if 'heads.' not in k
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
            if f'heads.{domain}' in k
        }


class ResNet18_DomainHeads(nn.Module):
    """ResNet18 with domain-specific classification heads.
    
    Architecture:
    - Backbone θ: ResNet18 conv1 → layer4 → avgpool (512-dim features)
    - Heads φ_e: One Linear(512, num_classes) per domain
    
    This follows the FedPer paradigm: shared backbone + personalized heads.
    """

    def __init__(
        self,
        num_classes: int = 126,
        domains: List[str] = None,
        pretrained: bool = True
    ):
        """Initialize ResNet18 with domain-specific heads.

        Args:
            num_classes: Number of output classes
            domains: List of domain names
            pretrained: Whether to load pretrained weights
        """
        super().__init__()
        
        if domains is None:
            domains = ['clipart', 'infograph', 'painting', 
                      'quickdraw', 'real', 'sketch']
        
        self.num_classes = num_classes
        self.domains = domains
        
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
        
        # Domain-specific heads
        self.heads = nn.ModuleDict({
            domain: nn.Linear(self._feature_dim, num_classes)
            for domain in domains
        })
        
        # Initialize heads
        for head in self.heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

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
        return self.heads[domain](features)

    def parameters_theta(self) -> List[nn.Parameter]:
        """Get backbone parameters (θ) for global aggregation.

        Returns:
            List of backbone parameters (excludes heads)
        """
        params = []
        for name, param in self.named_parameters():
            if 'heads.' not in name:
                params.append(param)
        return params

    def parameters_phi(self, domain: str) -> List[nn.Parameter]:
        """Get domain-specific head parameters (φ_e).

        Args:
            domain: Domain name

        Returns:
            List of parameters for the domain's head
        """
        return list(self.heads[domain].parameters())

    def state_dict_theta(self) -> Dict[str, torch.Tensor]:
        """Export backbone state dict.

        Returns:
            State dict containing only backbone parameters
        """
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if 'heads.' not in k
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
            if f'heads.{domain}' in k
        }
