"""
utils/evaluator.py
==================
课程要求 / Course Requirements:
    - Measure Performance ✅
    - Deep Learning / PyTorch

在 test set 上计算并打印:
    - Overall Accuracy
    - Per-class Accuracy
    - Macro F1 Score
    - sklearn classification_report
绘制并保存 Confusion Matrix 热力图 (seaborn) -> outputs/{name}_confusion_matrix.png
所有结果保存为 outputs/metrics.json
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # 无显示器环境也可保存图片
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from .trainer import get_device


# -----------------------------------------------------------------------------
# Confusion matrix plotting
# -----------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """绘制并保存归一化混淆矩阵热力图。"""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    model_name: str = "model",
    output_dir: str = "outputs",
    device: torch.device | None = None,
    save_to_json: bool = True,
) -> Dict:
    """
    Evaluate a trained model on the test set and dump metrics + confusion matrix.

    Returns
    -------
    metrics : dict
        Same data that gets persisted to outputs/metrics.json (cumulatively).
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------
    overall_acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class_acc = {}
    for i, name in enumerate(class_names):
        denom = cm[i].sum()
        per_class_acc[name] = float(cm[i, i] / denom) if denom > 0 else 0.0

    cls_report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"Evaluation report for: {model_name}")
    print("=" * 70)
    print(f"Overall Accuracy : {overall_acc*100:.2f}%")
    print(f"Macro F1 Score   : {macro_f1*100:.2f}%")
    print("\nPer-class Accuracy:")
    for name, acc in per_class_acc.items():
        print(f"  {name:<22s} : {acc*100:6.2f}%")
    print("\nClassification Report:")
    print(cls_report)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(
        cm, class_names, cm_path, title=f"{model_name} Confusion Matrix"
    )
    print(f"[Eval] Confusion matrix saved -> {cm_path}")

    metrics_for_model = {
        "model": model_name,
        "overall_accuracy": overall_acc,
        "macro_f1": macro_f1,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
    }

    if save_to_json:
        json_path = os.path.join(output_dir, "metrics.json")
        # 累积写入：保留其他模型已有的数据
        store: Dict[str, Dict] = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    store = json.load(f)
            except (json.JSONDecodeError, OSError):
                store = {}
        store[model_name] = metrics_for_model
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
        print(f"[Eval] Metrics appended to {json_path}")

    return metrics_for_model
