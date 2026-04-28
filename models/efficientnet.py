"""
models/efficientnet.py
======================
课程要求 / Course Requirements:
    - Transfer Learning  ✅
    - Deep Learning / PyTorch

加载 ImageNet 预训练 EfficientNet-B0, 替换 classifier 并支持三种 finetune 策略：
    - "full_freeze"   : 仅训练分类头
    - "partial"       : 解冻 features 最后两个 block + 分类头 (默认)
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


def build_efficientnet_b0(
    num_classes: int = 10,
    finetune_mode: FinetuneMode = "partial",
    pretrained: bool = True,
) -> nn.Module:
    """
    Build an EfficientNet-B0 (ImageNet pretrained) for EuroSAT.
    """
    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)

    _set_requires_grad(model, False)

    # classifier[1] 是最后的 Linear(1280, 1000)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )

    if finetune_mode == "full_freeze":
        pass
    elif finetune_mode == "partial":
        # EfficientNet-B0 features 共 9 个 sub-block (索引 0..8)
        # 解冻最后两个 (索引 7、8) 提供更强的特征适配能力
        _set_requires_grad(model.features[7], True)
        _set_requires_grad(model.features[8], True)
    elif finetune_mode == "full_finetune":
        _set_requires_grad(model, True)
    else:
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}")

    _set_requires_grad(model.classifier, True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[EfficientNet-B0] mode={finetune_mode}  "
        f"trainable={trainable:,}/{total:,} "
        f"({100*trainable/total:.2f}%)"
    )
    return model


if __name__ == "__main__":
    m = build_efficientnet_b0()
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    print(f"EfficientNet-B0 output: {y.shape}")
