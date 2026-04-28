"""
utils/visualizer.py
===================
课程要求 / Course Requirements:
    - Measure Performance ✅
    - Deep Learning / PyTorch

提供两个可视化函数:
    1. plot_training_curves(history, model_name)
        绘制 loss 和 accuracy 曲线 -> outputs/{model_name}_curves.png
    2. plot_model_comparison(results_dict)
        绘制多模型对比柱状图 (Baseline vs ResNet50 vs EfficientNet)
        -> outputs/model_comparison.png
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    output_dir: str = "outputs",
) -> str:
    """绘制训练 / 验证集的 loss 和 accuracy 曲线。"""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train", marker="o")
    ax.plot(epochs, history["val_loss"], label="val", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    # Accuracy
    ax = axes[1]
    ax.plot(
        epochs, [a * 100 for a in history["train_acc"]], label="train", marker="o"
    )
    ax.plot(
        epochs, [a * 100 for a in history["val_acc"]], label="val", marker="s"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model_name} — Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_curves.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Visualizer] training curves saved -> {save_path}")
    return save_path


def plot_model_comparison(
    results: Dict[str, Dict],
    output_dir: str = "outputs",
) -> str:
    """
    绘制 Accuracy / F1 对比柱状图.

    Parameters
    ----------
    results : dict
        Mapping ``model_name -> {"overall_accuracy": float, "macro_f1": float}``.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not results:
        print("[Visualizer] no results provided, skipping comparison plot.")
        return ""

    model_names = list(results.keys())
    accs = [results[m].get("overall_accuracy", 0.0) * 100 for m in model_names]
    f1s = [results[m].get("macro_f1", 0.0) * 100 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy (%)", color="#00d4aa")
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1 (%)", color="#4f8cff")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=0)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Model Comparison on EuroSAT (Test Set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.8,
                f"{b.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    save_path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Visualizer] comparison chart saved -> {save_path}")
    return save_path
