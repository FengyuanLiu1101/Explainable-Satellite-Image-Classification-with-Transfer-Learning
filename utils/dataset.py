"""
utils/dataset.py
================
课程要求 / Course Requirements:
    - Deep Learning / PyTorch
    - Data preparation pipeline for the EuroSAT (RGB) land-use dataset.

功能 / Function:
    1. 使用 torchvision.datasets.ImageFolder 加载 EuroSAT (10 classes)
    2. 70% / 15% / 15%  train / val / test split, seed = 42
    3. Train  : RandomHorizontalFlip + RandomVerticalFlip
                + RandomRotation(15) + ColorJitter
       Val/Test: Resize(64x64) + Normalize 仅做归一化
    4. EuroSAT RGB statistics:
           mean = [0.3444, 0.3803, 0.4078]
           std  = [0.2037, 0.1366, 0.1148]
    5. 返回 dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# -----------------------------------------------------------------------------
# 全局常量 / Global constants
# -----------------------------------------------------------------------------
SEED: int = 42

EUROSAT_MEAN: Tuple[float, float, float] = (0.3444, 0.3803, 0.4078)
EUROSAT_STD: Tuple[float, float, float] = (0.2037, 0.1366, 0.1148)

# EuroSAT 10 类 (按 ImageFolder 字母序自动得出，此处仅供参考显示)
EUROSAT_CLASSES: List[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# 类别 emoji 图标，用于 Web Demo 前端展示
EUROSAT_EMOJI: Dict[str, str] = {
    "AnnualCrop": "🌾",
    "Forest": "🌳",
    "HerbaceousVegetation": "🌿",
    "Highway": "🛣️",
    "Industrial": "🏭",
    "Pasture": "🐄",
    "PermanentCrop": "🌳",
    "Residential": "🏘️",
    "River": "🌊",
    "SeaLake": "🌊",
}


def set_seed(seed: int = SEED) -> None:
    """统一随机种子，保证实验可复现 / Reproducibility helper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def _train_transform() -> transforms.Compose:
    """训练集的数据增强 / Training-time augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
        ]
    )


def _eval_transform() -> transforms.Compose:
    """验证 / 测试集的预处理 / Eval-time transform (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD),
        ]
    )


def get_inference_transform() -> transforms.Compose:
    """供推理 / Web Demo 使用，确保和 val/test 一致。"""
    return _eval_transform()


# -----------------------------------------------------------------------------
# Subset wrapper 让 train / val / test 可以使用不同 transform
# -----------------------------------------------------------------------------
class _TransformedSubset(torch.utils.data.Dataset):
    """Wrapper that applies a transform on top of a torch Subset."""

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform
        # 透传 ImageFolder 的类别信息
        base = subset.dataset
        self.classes = getattr(base, "classes", EUROSAT_CLASSES)
        self.class_to_idx = getattr(base, "class_to_idx", None)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        # ImageFolder 的内部 transform 已置 None，这里手动应用
        path, target = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image  # 延迟导入，避免不必要的依赖
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


# -----------------------------------------------------------------------------
# 主 API / Public API
# -----------------------------------------------------------------------------
def build_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = SEED,
) -> Dict[str, DataLoader]:
    """
    Build {train, val, test} DataLoaders for EuroSAT.

    Parameters
    ----------
    data_dir : str
        EuroSAT 根目录 (须含 10 个子文件夹)
    batch_size : int
        Mini-batch size
    num_workers : int
        DataLoader worker 数量
    seed : int
        随机种子

    Returns
    -------
    Dict[str, DataLoader]
        {"train": ..., "val": ..., "test": ...}
    """
    set_seed(seed)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"EuroSAT data directory not found: {data_dir}\n"
            f"请先下载 EuroSAT 数据集并解压到 {data_dir}"
        )

    # 不在 ImageFolder 上挂 transform，便于按 split 应用不同 transform
    full_dataset = datasets.ImageFolder(root=data_dir, transform=None)

    n_total = len(full_dataset)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    train_ds = _TransformedSubset(train_subset, _train_transform())
    val_ds = _TransformedSubset(val_subset, _eval_transform())
    test_ds = _TransformedSubset(test_subset, _eval_transform())

    pin = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
    }

    print(
        f"[Dataset] EuroSAT loaded: total={n_total} | "
        f"train={n_train}  val={n_val}  test={n_test} | "
        f"classes={len(full_dataset.classes)}"
    )
    return loaders


if __name__ == "__main__":
    # 简单的自测脚本 / Self-test
    loaders = build_dataloaders("./data/EuroSAT", batch_size=8, num_workers=0)
    x, y = next(iter(loaders["train"]))
    print("train batch:", x.shape, y.shape, "dtype:", x.dtype)
