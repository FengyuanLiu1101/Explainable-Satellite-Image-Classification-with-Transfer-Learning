"""
models/resnet50.py
==================
Course Requirements:
    - Transfer Learning  ✅
    - Deep Learning / PyTorch

Loads an ImageNet-pretrained ResNet-50, replaces its classification head,
and supports three fine-tuning strategies:
    - "full_freeze"   : Freeze the entire backbone; only the new classifier head is trained.
    - "partial"       : Unfreeze layer3 + layer4 + classifier head (default strategy).
    - "full_finetune" : All parameters participate in training (no freezing).
"""

from typing import Literal
import torch
import torch.nn as nn
import torchvision.models as tvm

# ─────────────────────────────────────────────────────────────────────────────
# Type alias for the three supported fine-tuning modes.
# Using Literal restricts the argument to exactly these three string values,
# so passing anything else raises a type-checking error at development time.
# ─────────────────────────────────────────────────────────────────────────────
FinetuneMode = Literal["full_freeze", "partial", "full_finetune"]


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Enable or disable gradient computation for every parameter in a module.

    How it works
    ------------
    PyTorch tracks whether a parameter should accumulate gradients via its
    `requires_grad` flag. When `requires_grad=False`, the autograd engine
    skips that parameter entirely during backpropagation — this is what
    "freezing" a layer means in practice.

    Parameters
    ----------
    module : nn.Module
        Any PyTorch module (a single layer, a block, or the whole model).
    requires_grad : bool
        True  → the parameter will be updated during training (unfrozen).
        False → the parameter is frozen and will NOT be updated.
    """
    for p in module.parameters():
        p.requires_grad = requires_grad


def build_resnet50(
    num_classes: int = 10,
    finetune_mode: FinetuneMode = "partial",
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a ResNet-50 model with ImageNet pretrained weights, adapted for
    a custom classification task (e.g., EuroSAT with 10 land-use classes).

    Transfer Learning Strategy
    --------------------------
    Instead of training from scratch (which would require enormous data and
    compute), we start from weights that already encode rich visual features
    learned on 1.28 million ImageNet images.  We then adapt the network to
    our target domain by:
      1. Replacing the original 1000-class head with a new head sized for
         our task.
      2. Selectively unfreezing layers depending on `finetune_mode`.

    Parameters
    ----------
    num_classes : int
        Number of output classes for the new task (10 for EuroSAT).
    finetune_mode : FinetuneMode
        Controls which layers are trainable — see detailed notes below.
    pretrained : bool
        If True, load ImageNet1K_V2 weights (higher accuracy than V1).
        If False, all weights are randomly initialised (useful for ablation
        studies to confirm that pretraining is actually helping).

    Returns
    -------
    nn.Module
        The modified ResNet-50 ready for training.

    Fine-tuning Mode Details
    -------------------------
    "full_freeze"   — Only the new classification head trains.
                      The backbone acts as a fixed feature extractor.
                      • Fastest training, lowest memory.
                      • Best when your dataset is very small or very similar
                        to ImageNet.

    "partial"       — layer3, layer4, and the head are trainable; everything
                      before layer3 (conv1, bn1, layer1, layer2) stays frozen.
                      • Good balance of speed and adaptability.
                      • Default choice for most fine-tuning tasks.

    "full_finetune" — Every single parameter is unfrozen.
                      • Slowest, highest memory, but highest potential accuracy
                        when you have sufficient data.
                      • Risk of catastrophic forgetting if learning rate is
                        too large; use a small LR with a scheduler.
    """

    # ── Step 0: Load the pretrained backbone ──────────────────────────────────
    # IMAGENET1K_V2 uses better training recipes (larger crops, label smoothing,
    # MixUp) than V1, giving ~1% higher top-1 accuracy out of the box.
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tvm.resnet50(weights=weights)

    # ── Step 1: Freeze ALL parameters as the default baseline ─────────────────
    # We start by freezing everything. This makes the logic for the three modes
    # cleaner: each mode only needs to *unfreeze* the layers it cares about,
    # rather than freeze layers it doesn't care about.
    _set_requires_grad(model, False)

    # ── Step 2: Replace the classification head ───────────────────────────────
    # The original ResNet-50 head is a single Linear(2048 → 1000) layer.
    # We replace it with a small MLP that:
    #   • Uses Dropout to regularise and reduce overfitting.
    #   • Projects from 2048 → 256 with a hidden layer so the network can
    #     learn a more complex decision boundary.
    #   • Outputs `num_classes` logits (raw scores before softmax).
    #
    # Because this is a brand-new layer (randomly initialised), it MUST be
    # trainable regardless of the fine-tuning mode — hence we re-enable its
    # gradients unconditionally at the end of Step 3.
    in_features = model.fc.in_features  # Always 2048 for ResNet-50
    model.fc = nn.Sequential(
        nn.Dropout(0.4),                        # Aggressive dropout before first linear
        nn.Linear(in_features, 256),            # Bottleneck: 2048 → 256
        nn.ReLU(inplace=True),                  # Non-linearity; inplace saves memory
        nn.Dropout(0.2),                        # Lighter dropout before output layer
        nn.Linear(256, num_classes),            # Final projection → class logits
    )

    # ── Step 3: Selectively unfreeze layers per fine-tuning mode ──────────────
    if finetune_mode == "full_freeze":
        # Nothing extra to unfreeze — only the head (Step 4) will train.
        pass

    elif finetune_mode == "partial":
        # Unfreeze the two deepest residual stages of the backbone.
        #
        # ResNet-50 architecture recap:
        #   conv1 → bn1 → relu → maxpool
        #   → layer1 (3 bottleneck blocks, 256-ch feature maps)
        #   → layer2 (4 bottleneck blocks, 512-ch feature maps)
        #   → layer3 (6 bottleneck blocks, 1024-ch feature maps)  ← unfreeze
        #   → layer4 (3 bottleneck blocks, 2048-ch feature maps)  ← unfreeze
        #   → avgpool → fc (our new head)
        #
        # layer3 and layer4 encode high-level semantics (shapes, textures,
        # object parts) that are most likely to differ between ImageNet and
        # your target domain, so adapting them usually gives the biggest gain.
        _set_requires_grad(model.layer3, True)
        _set_requires_grad(model.layer4, True)

    elif finetune_mode == "full_finetune":
        # Re-enable gradients for the entire network, overriding Step 1.
        _set_requires_grad(model, True)

    else:
        # Guard against typos or unsupported modes passed at runtime.
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}")

    # ── Step 4: Ensure the new head is ALWAYS trainable ───────────────────────
    # This line is critical for "full_freeze" mode: even though the backbone is
    # completely frozen, the randomly-initialised head MUST update or the model
    # cannot learn anything at all.
    # For "partial" and "full_finetune" it is redundant but harmless.
    _set_requires_grad(model.fc, True)

    # ── Step 5: Diagnostic summary ────────────────────────────────────────────
    # Print how many parameters are actually trainable so you can sanity-check
    # that the freezing/unfreezing logic did what you expected.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"[ResNet-50] mode={finetune_mode}  "
        f"trainable={trainable:,}/{total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test: run this file directly to verify the model builds and
# produces output tensors of the expected shape.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build model with default settings (partial fine-tuning, 10 classes)
    m = build_resnet50()

    # Create a random batch: 2 images, 3 colour channels, 64×64 pixels
    # (EuroSAT images are 64×64; ResNet-50 works with any spatial resolution
    #  ≥ 32×32 due to the adaptive average pooling layer before the head)
    x = torch.randn(2, 3, 64, 64)

    # Forward pass — should output shape [2, 10]
    y = m(x)
    print(f"ResNet-50 output: {y.shape}")  # Expected: torch.Size([2, 10])
