"""Single-head ResNet50 for FedAvg/FedProx baseline experiments.

This model differs from ResNet50_EAPH in that:
- No LoRA modules (no domain adaptation)
- Single global classification head (no domain-specific heads)
- Simpler interface for standard FedAvg aggregation
"""

from typing import Dict, List
import torch
import torch.nn as nn
from torchvision import models


class ResNet50Single(nn.Module):
    """Single-head ResNet50 for global federated learning.

    A standard ResNet-50 model with a single classification head,
    used for FedAvg and FedProx baseline experiments.
    """

    def __init__(
        self,
        num_classes: int = 126,
        pretrained: bool = True
    ):
        """Initialize ResNet50 with single global head.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to load pretrained ImageNet weights
        """
        super().__init__()

        self.num_classes = num_classes

        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)

        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Feature dimension from ResNet50
        self._feature_dim = 2048

        # Single global classification head
        self.fc = nn.Linear(self._feature_dim, num_classes)

        # Initialize head
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    @property
    def feature_dim(self) -> int:
        """Get feature dimension for compatibility.

        Returns:
            Feature dimension (2048 for ResNet50)
        """
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head.

        Args:
            x: Input tensor of shape (N, 3, H, W)

        Returns:
            2048-dimensional feature vector of shape (N, 2048)
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
