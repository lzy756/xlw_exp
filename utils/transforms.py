"""Image transforms for DomainNet dataset."""

from torchvision import transforms


def build_transforms(train: bool = True) -> transforms.Compose:
    """Build image transformation pipeline.

    Args:
        train: Whether to include training augmentations

    Returns:
        Composed transformation pipeline
    """
    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Validation transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    return transform