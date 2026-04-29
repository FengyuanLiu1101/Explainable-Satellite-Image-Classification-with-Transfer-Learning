"""
models/resnet50.py
==================
课程要求 / Course Requirements:
    - Transfer Learning  ✅
    - Deep Learning / PyTorch

加载 ImageNet 预训练 ResNet-50, 替换分类头并支持三种 finetune 策略：
    - "full_freeze"   : 冻结所有 backbone, 仅训练分类头
    - "partial"       : 解冻 layer3 + layer4 + 分类头 (默认)
    - "full_finetune" : 全部参数参与训练
"""

from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as tvm

FinetuneMode = Literal["full_freeze", "partial", "full_finetune"]


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad


def build_resnet50(
    num_classes: int = 10,
    finetune_mode: FinetuneMode = "partial",
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a ResNet-50 with ImageNet pretrained weights for EuroSAT.

    Parameters
    ----------
    num_classes : int
        Number of output classes (10 for EuroSAT).
    finetune_mode : str
        "full_freeze" / "partial" / "full_finetune"
    pretrained : bool
        Whether to load ImageNet1K_V2 weights.
    """
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tvm.resnet50(weights=weights)

    # 1) 先冻结所有参数
    _set_requires_grad(model, False)

    # 2) 替换分类头 (always trainable)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )

    # 3) 根据 finetune_mode 解冻
    if finetune_mode == "full_freeze":
        pass  # 仅分类头可训练
    elif finetune_mode == "partial":
        _set_requires_grad(model.layer3, True)
        _set_requires_grad(model.layer4, True)
    elif finetune_mode == "full_finetune":
        _set_requires_grad(model, True)
    else:
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}")

    # 分类头始终可训
    _set_requires_grad(model.fc, True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[ResNet-50] mode={finetune_mode}  "
        f"trainable={trainable:,}/{total:,} "
        f"({100*trainable/total:.2f}%)"
    )
    return model


if __name__ == "__main__":
    m = build_resnet50()
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    print(f"ResNet-50 output: {y.shape}")
