"""
evaluate.py
===========
课程要求 / Course Requirements:
    - Measure Performance ✅
    - Deep Learning / PyTorch ✅

CLI:
    python evaluate.py --model [baseline|resnet50|efficientnet]
                       --weights outputs/resnet50_best.pth
                       --data_dir ./data/EuroSAT
"""

from __future__ import annotations

import argparse
import os

import torch

from models import build_model
from utils.dataset import EUROSAT_CLASSES, build_dataloaders
from utils.evaluator import evaluate_model
from utils.trainer import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on EuroSAT")
    parser.add_argument(
        "--model",
        required=True,
        choices=["baseline", "resnet50", "efficientnet"],
    )
    parser.add_argument("--weights", required=True, help="Path to .pth")
    parser.add_argument("--data_dir", default="./data/EuroSAT")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument(
        "--mode",
        default="partial",
        choices=["full_freeze", "partial", "full_finetune"],
        help="Should match how the checkpoint was trained.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    loaders = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, num_classes=10, finetune_mode=args.mode)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    evaluate_model(
        model=model,
        test_loader=loaders["test"],
        class_names=EUROSAT_CLASSES,
        model_name=args.model,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
