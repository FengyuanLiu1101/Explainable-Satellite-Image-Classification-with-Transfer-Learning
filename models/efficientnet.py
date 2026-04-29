"""
models/efficientnet.py
======================
Course Requirements:
    - Transfer Learning  ✅
    - Deep Learning / PyTorch

Loads an ImageNet-pretrained EfficientNet-B0, replaces the classifier head,
and supports three fine-tuning strategies:
    - "full_freeze"   : Freeze the entire backbone; only train the classifier head.
    - "partial"       : Unfreeze the last two feature blocks + classifier head (default).
    - "full_finetune" : All parameters participate in training.
"""

from typing import Literal
import torch
import torch.nn as nn
import torchvision.models as tvm

# ─────────────────────────────────────────────────────────────────────────────
# Type alias restricting finetune_mode to exactly these three string values.
# Passing any other string will raise a type error at development time.
# ─────────────────────────────────────────────────────────────────────────────
FinetuneMode = Literal["full_freeze", "partial", "full_finetune"]


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Enable or disable gradient computation for every parameter in a module.

    This is the core mechanism behind "freezing" and "unfreezing" layers.
    When requires_grad=False, PyTorch's autograd engine skips that parameter
    entirely during backpropagation — no gradient is computed, and the
    optimiser makes no update to that parameter.

    Parameters
    ----------
    module : nn.Module
        Any PyTorch module — a single layer, a block, or the entire model.
    requires_grad : bool
        True  → parameter is unfrozen and will be updated during training.
        False → parameter is frozen and will NOT be updated.
    """
    for p in module.parameters():
        p.requires_grad = requires_grad


def build_efficientnet_b0(
    num_classes: int = 10,
    finetune_mode: FinetuneMode = "partial",
    pretrained: bool = True,
) -> nn.Module:
    """
    Build an EfficientNet-B0 with ImageNet pretrained weights, adapted for
    a custom classification task (e.g., EuroSAT with 10 land-use classes).

    How EfficientNet-B0 differs from ResNet-50
    -------------------------------------------
    While ResNet-50 scales depth only, EfficientNet uses "compound scaling" —
    it balances depth, width, and input resolution simultaneously.
    EfficientNet-B0 is the smallest baseline model in the EfficientNet family,
    yet achieves comparable accuracy to ResNet-50 with far fewer parameters:
        EfficientNet-B0 : ~5.3M parameters
        ResNet-50        : ~25.6M parameters

    EfficientNet-B0 Architecture
    -----------------------------
    The backbone is organised into a `features` sequential container
    with 9 sub-blocks (index 0 to 8):

        features[0]  — Initial stem conv (3 → 32, stride 2)
        features[1]  — MBConv1  block (32 → 16)
        features[2]  — MBConv6  block (16 → 24)  x2
        features[3]  — MBConv6  block (24 → 40)  x2
        features[4]  — MBConv6  block (40 → 80)  x3
        features[5]  — MBConv6  block (80 → 112) x3
        features[6]  — MBConv6  block (112 → 192) x4
        features[7]  — MBConv6  block (192 → 320) x1   ← unfrozen in "partial"
        features[8]  — Final conv (320 → 1280)          ← unfrozen in "partial"
        classifier   — Dropout + Linear(1280 → 10)      ← always trainable

    MBConv = Mobile Inverted Bottleneck Convolution, the core building block
    of EfficientNet. Each block also uses Squeeze-and-Excitation (SE) attention
    to recalibrate channel-wise feature responses.

    Parameters
    ----------
    num_classes : int
        Number of output classes for the new task (10 for EuroSAT).
    finetune_mode : FinetuneMode
        Controls which layers are trainable — see detailed notes below.
    pretrained : bool
        If True, load ImageNet1K_V1 weights (pretrained on 1.28M images).
        If False, all weights are randomly initialised.

    Fine-tuning Mode Details
    -------------------------
    "full_freeze"   — Only the new classifier head trains.
                      The entire feature extractor is locked.
                      • Fastest, lowest memory.
                      • Best for very small datasets.

    "partial"       — features[7], features[8], and the classifier train.
                      Everything before features[7] stays frozen.
                      • Default and recommended choice for most tasks.
                      • Adapts the highest-level semantic features while
                        preserving the general low-level representations.

    "full_finetune" — Every parameter is unfrozen and updated.
                      • Highest potential accuracy, but slowest and most
                        prone to overfitting on small datasets.
                      • Requires a small learning rate to avoid destroying
                        the pretrained weights (catastrophic forgetting).

    Returns
    -------
    nn.Module
        The modified EfficientNet-B0 ready for training.
    """

    # ── Step 0: Load the pretrained backbone ──────────────────────────────────
    # EfficientNet-B0 only has IMAGENET1K_V1 weights available in torchvision
    # (unlike ResNet-50 which has the improved V2 weights).
    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)

    # ── Step 1: Freeze ALL parameters as the default baseline ─────────────────
    # Start by freezing everything. Each fine-tuning mode then selectively
    # unfreezes only the layers it needs — cleaner than the reverse approach.
    _set_requires_grad(model, False)

    # ── Step 2: Replace the classification head ───────────────────────────────
    # The original EfficientNet-B0 classifier is:
    #     Sequential(Dropout(0.2), Linear(1280 → 1000))
    #
    # We replace it with a simpler head for our task:
    #     Sequential(Dropout(0.4), Linear(1280 → num_classes))
    #
    # Key difference vs ResNet-50 head:
    #   - EfficientNet already has a strong 1280-d representation after
    #     the final conv, so a single linear layer is sufficient here.
    #   - ResNet-50 used a two-layer MLP (2048→256→10); EfficientNet uses
    #     a direct projection (1280→10) — simpler but equally effective.
    #
    # The new head is randomly initialised and MUST be trainable, so its
    # gradients are re-enabled unconditionally in Step 4.
    in_features = model.classifier[1].in_features  # Always 1280 for EfficientNet-B0
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),                   # Regularisation to reduce overfitting
        nn.Linear(in_features, num_classes),  # Direct projection to class logits
    )

    # ── Step 3: Selectively unfreeze layers per fine-tuning mode ──────────────
    if finetune_mode == "full_freeze":
        # Nothing extra to unfreeze — only the classifier head (Step 4) trains.
        pass

    elif finetune_mode == "partial":
        # Unfreeze the last two sub-blocks of the feature extractor.
        #
        # features[7]: MBConv6 block (192 → 320)
        #   — Encodes the highest-level semantic representations before the
        #     final conv. Most likely to differ between ImageNet and EuroSAT.
        #
        # features[8]: Final conv + BN (320 → 1280)
        #   — Aggregates all spatial features into a rich 1280-d descriptor.
        #     Fine-tuning this directly improves the quality of the embedding
        #     fed into the classifier.
        _set_requires_grad(model.features[7], True)
        _set_requires_grad(model.features[8], True)

    elif finetune_mode == "full_finetune":
        # Re-enable gradients for the entire network, overriding Step 1.
        _set_requires_grad(model, True)

    else:
        # Guard against unsupported mode strings passed at runtime.
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}")

    # ── Step 4: Ensure the classifier head is ALWAYS trainable ────────────────
    # Critical for "full_freeze" mode: the randomly-initialised head MUST
    # update or the model cannot learn anything at all.
    _set_requires_grad(model.classifier, True)

    # ── Step 5: Diagnostic summary ────────────────────────────────────────────
    # Print trainable vs total parameters to verify the freezing logic worked
    # as expected before committing to a full training run.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"[EfficientNet-B0] mode={finetune_mode}  "
        f"trainable={trainable:,}/{total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test: run this file directly to confirm the model builds and
# produces output tensors of the expected shape.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build model with default settings (partial fine-tuning, 10 classes)
    m = build_efficientnet_b0()

    # Create a random batch: 2 images, 3 colour channels, 64×64 pixels
    x = torch.randn(2, 3, 64, 64)

    # Forward pass — should output shape [2, 10]
    y = m(x)
    print(f"EfficientNet-B0 output: {y.shape}")  # Expected: torch.Size([2, 10])
