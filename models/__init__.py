"""
Models package — three architectures used in this project:
    - BaselineCNN  : 从零训练 (control group)
    - ResNet-50    : Transfer Learning (ImageNet pretrained)
    - EfficientNetB0: Transfer Learning (ImageNet pretrained)
"""

from .baseline_cnn import BaselineCNN, build_baseline_cnn
from .resnet50 import build_resnet50
from .efficientnet import build_efficientnet_b0


def build_model(name: str, num_classes: int = 10, finetune_mode: str = "partial"):
    """
    Factory that returns the requested model.

    name in {"baseline", "resnet50", "efficientnet"}
    """
    name = name.lower()
    if name == "baseline":
        return build_baseline_cnn(num_classes=num_classes)
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, finetune_mode=finetune_mode)
    if name == "efficientnet":
        return build_efficientnet_b0(num_classes=num_classes, finetune_mode=finetune_mode)
    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "BaselineCNN",
    "build_baseline_cnn",
    "build_resnet50",
    "build_efficientnet_b0",
    "build_model",
]
