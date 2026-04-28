"""
utils/trainer.py
================
课程要求 / Course Requirements:
    - Deep Learning / PyTorch
    - Measure Performance (训练日志 + 最佳权重保存)

通用训练循环 / Generic training loop with:
    - Optimizer        : AdamW (lr=1e-4, weight_decay=1e-4)
    - Loss             : CrossEntropyLoss(label_smoothing=0.1)
    - LR scheduler     : CosineAnnealingLR
    - 自动保存 val_acc 最高的权重 -> outputs/<name>_best.pth
    - 设备自动检测 (CUDA / MPS / CPU)
    - 日志同时打印到控制台 + outputs/train.log
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
# Device / Logging helpers
# -----------------------------------------------------------------------------
def get_device() -> torch.device:
    """自动选择最佳设备 / Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _setup_logger(log_path: str) -> logging.Logger:
    """同时输出到控制台和文件 / Console + file logger."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("eurosat_trainer")
    logger.setLevel(logging.INFO)
    # 防止重复 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# -----------------------------------------------------------------------------
# Train / Validate one epoch
# -----------------------------------------------------------------------------
def _run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
):
    """Run one epoch (train or eval) and return (avg_loss, avg_acc)."""
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    model_name: str,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    output_dir: str = "outputs",
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """
    Train ``model`` and return its training history.

    Returns
    -------
    history : dict of lists
        Keys: train_loss / train_acc / val_loss / val_acc
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    logger = _setup_logger(log_path)

    device = device or get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    # 仅对 requires_grad 的参数构建 optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    best_path = os.path.join(output_dir, f"{model_name}_best.pth")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    logger.info("=" * 70)
    logger.info(
        f"Training '{model_name}' on {device} | epochs={epochs} | "
        f"lr={lr} | wd={weight_decay} | smoothing={label_smoothing}"
    )
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = _run_one_epoch(
            model, loaders["train"], criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_one_epoch(
            model, loaders["val"], criterion, None, device, train=False
        )
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"[{model_name}] Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} acc={val_acc*100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            logger.info(
                f"[{model_name}] ✓ saved new best to {best_path} "
                f"(val_acc={val_acc*100:.2f}%)"
            )

    logger.info(
        f"[{model_name}] training done. best_val_acc={best_val_acc*100:.2f}%"
    )
    return history
