"""
train.py
========
课程要求 / Course Requirements:
    - Transfer Learning ✅
    - Deep Learning / PyTorch ✅
    - Measure Performance ✅

主训练入口 / Main entry-point.

CLI:
    python train.py --model [baseline|resnet50|efficientnet|all]  (default: all)
                    --epochs 20
                    --data_dir ./data/EuroSAT
                    --mode [full_freeze|partial|full_finetune]    (default: partial)

运行 ``--model all`` 时, 依次训练三种模型并在结尾生成对比图 model_comparison.png.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch

from models import build_model
from utils.dataset import EUROSAT_CLASSES, build_dataloaders, set_seed
from utils.evaluator import evaluate_model
from utils.trainer import get_device, train_model
from utils.visualizer import plot_model_comparison, plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EuroSAT classifiers")
    parser.add_argument(
        "--model",
        default="all",
        choices=["baseline", "resnet50", "efficientnet", "all"],
        help="Which model to train. 'all' trains the three sequentially.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_dir", default="./data/EuroSAT")
    parser.add_argument(
        "--mode",
        default="partial",
        choices=["full_freeze", "partial", "full_finetune"],
        help="Finetune strategy for transfer-learning models.",
    )
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one(
    model_name: str,
    args: argparse.Namespace,
    loaders: Dict,
    device: torch.device,
) -> Dict:
    """Train + evaluate a single model. Returns its test metrics dict."""
    print("\n" + "#" * 70)
    print(f"#   Training {model_name}  (mode={args.mode})")
    print("#" * 70)

    model = build_model(model_name, num_classes=10, finetune_mode=args.mode)
    history = train_model(
        model=model,
        loaders=loaders,
        model_name=model_name,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        device=device,
    )
    plot_training_curves(history, model_name, output_dir=args.output_dir)

    # Reload best weights before evaluation
    best_path = os.path.join(args.output_dir, f"{model_name}_best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[Train] reloaded best checkpoint: {best_path}")

    metrics = evaluate_model(
        model=model,
        test_loader=loaders["test"],
        class_names=EUROSAT_CLASSES,
        model_name=model_name,
        output_dir=args.output_dir,
        device=device,
    )
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"[Train] device = {device}")

    loaders = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if args.model == "all":
        targets: List[str] = ["baseline", "resnet50", "efficientnet"]
    else:
        targets = [args.model]

    results: Dict[str, Dict] = {}
    for name in targets:
        results[name] = train_one(name, args, loaders, device)

    if len(results) >= 2:
        plot_model_comparison(results, output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("Final Test Results")
    print("=" * 70)
    print(f"{'Model':<15s} {'Accuracy':>10s} {'Macro F1':>10s}")
    for name, m in results.items():
        print(
            f"{name:<15s} {m['overall_accuracy']*100:>9.2f}% "
            f"{m['macro_f1']*100:>9.2f}%"
        )


if __name__ == "__main__":
    main()
