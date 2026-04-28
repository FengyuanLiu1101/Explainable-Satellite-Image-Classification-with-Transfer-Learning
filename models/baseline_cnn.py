"""
models/baseline_cnn.py
======================
课程要求 / Course Requirements:
    - Deep Learning / PyTorch
    - 对照组：从零训练的 4 层 CNN (control group, no transfer learning)
      用于证明迁移学习的价值 (Transfer Learning value justification)

结构 / Architecture:
    Conv → BN → ReLU → MaxPool   ×4
    AdaptiveAvgPool(1) → Flatten → FC

输入 / Input  shape : (B, 3, 64, 64)
输出 / Output shape : (B, 10)
"""

import torch
import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class BaselineCNN(nn.Module):
    """4-layer convolutional baseline (trained from scratch)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),    # 64 -> 32
            _conv_block(32, 64),   # 32 -> 16
            _conv_block(64, 128),  # 16 ->  8
            _conv_block(128, 256), #  8 ->  4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_baseline_cnn(num_classes: int = 10) -> BaselineCNN:
    """Factory function for use by train.py / evaluate.py."""
    return BaselineCNN(num_classes=num_classes)


if __name__ == "__main__":
    m = build_baseline_cnn()
    x = torch.randn(2, 3, 64, 64)
    y = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"BaselineCNN output={y.shape}  params={n_params:,}")
