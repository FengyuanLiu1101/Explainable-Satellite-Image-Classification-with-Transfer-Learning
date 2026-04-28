"""
gradcam.py
==========
课程要求 / Course Requirements:
    - Deep Learning / PyTorch
    - Measure Performance / Visualization

手动实现 GradCAM (不依赖 third-party grad-cam 库)：
    1. 通过 forward hook 记录目标层的 feature map
    2. 通过 backward hook 记录目标层对应的梯度
    3. 全局平均池化梯度得到通道权重 -> 加权特征图 -> ReLU -> 上采样
    4. 与原图融合输出热力图

提供:
    - GradCAM         : 类
    - generate_gradcam: 单张图便捷函数
    - batch_gradcam   : 每个类别取 N 张测试图生成热力图
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.dataset import (
    EUROSAT_CLASSES,
    EUROSAT_MEAN,
    EUROSAT_STD,
    get_inference_transform,
)
from utils.trainer import get_device


# -----------------------------------------------------------------------------
# Core class
# -----------------------------------------------------------------------------
class GradCAM:
    """
    Standard GradCAM. Hooks 在 __init__ 中注册, 使用完务必调用 ``remove_hooks``
    或在 with 语句中使用以确保资源释放。
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: list = []

        # 前向 hook: 记录 feature map
        self._handles.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        # 反向 hook: 记录梯度 (推荐 register_full_backward_hook)
        self._handles.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    # ------------------------- hooks ----------------------------------
    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        # grad_out 是 tuple, 第一项是相对该层输出的梯度
        self._gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    # ------------------------- compute --------------------------------
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute the GradCAM heatmap for ``input_tensor``.

        Parameters
        ----------
        input_tensor : (1, 3, H, W) tensor (already normalized)
        class_idx    : int, target class. If None, uses argmax prediction.

        Returns
        -------
        cam      : np.ndarray of shape (H, W) in [0, 1]
        pred_idx : int
        prob     : float (softmax probability of the target class)
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[:, class_idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "GradCAM hooks did not fire — check the target_layer."
            )

        # weights: (1, C, 1, 1)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        # Normalize 0..1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        prob = float(probs[0, class_idx].item())
        return cam, class_idx, prob


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalized tensor -> (H,W,3) uint8 RGB image."""
    mean = torch.tensor(EUROSAT_MEAN).view(3, 1, 1)
    std = torch.tensor(EUROSAT_STD).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def overlay_heatmap(
    rgb_img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """将 cam (H,W) ∈[0,1] 与 rgb_img (H,W,3) 融合, 返回 RGB uint8."""
    if cam.shape != rgb_img.shape[:2]:
        cam = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * rgb_img).astype(np.uint8)
    return overlay


def _get_default_target_layer(model: nn.Module) -> nn.Module:
    """
    自动找到默认的目标层:
        - ResNet            -> model.layer4[-1]
        - EfficientNet      -> model.features[-1]
        - BaselineCNN       -> features[-1] 中的最后一个 Conv
    """
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        feats = model.features
        # EfficientNet: features 是 Sequential
        return feats[-1]
    raise ValueError("Cannot infer GradCAM target layer for this model.")


# -----------------------------------------------------------------------------
# Convenience APIs
# -----------------------------------------------------------------------------
def generate_gradcam(
    model: nn.Module,
    image_path: str,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, int, float]:
    """
    生成单张图的 GradCAM 对比图 (原图 | 热力图 | 叠加图).

    Returns
    -------
    overlay : np.ndarray  (H, W, 3) uint8 RGB
    pred_idx: int
    prob    : float
    """
    device = device or get_device()
    model = model.to(device).eval()
    target_layer = target_layer or _get_default_target_layer(model)

    pil = Image.open(image_path).convert("RGB")
    tfm = get_inference_transform()
    x = tfm(pil).unsqueeze(0).to(device)

    with GradCAM(model, target_layer) as cam_extractor:
        cam, pred_idx, prob = cam_extractor(x, class_idx=target_class)

    rgb = _denormalize(x[0])
    overlay = overlay_heatmap(rgb, cam)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # 横向拼接：原图 | 热力图 | 叠加图
        heatmap_only = overlay_heatmap(np.zeros_like(rgb), cam)
        combined = np.concatenate([rgb, heatmap_only, overlay], axis=1)
        Image.fromarray(combined).save(save_path)

    return overlay, pred_idx, prob


def batch_gradcam(
    model: nn.Module,
    test_loader,
    output_dir: str = "outputs/gradcam",
    samples_per_class: int = 3,
    class_names: Iterable[str] = EUROSAT_CLASSES,
    target_layer: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> List[str]:
    """
    每个类别取 ``samples_per_class`` 张测试图生成 GradCAM 热力图.
    输入是 DataLoader (val/test loader from build_dataloaders).
    """
    device = device or get_device()
    model = model.to(device).eval()
    target_layer = target_layer or _get_default_target_layer(model)
    os.makedirs(output_dir, exist_ok=True)

    class_names = list(class_names)
    counters = {i: 0 for i in range(len(class_names))}
    saved_paths: List[str] = []

    cam_extractor = GradCAM(model, target_layer)
    try:
        for batch_x, batch_y in test_loader:
            for i in range(batch_x.size(0)):
                cls = int(batch_y[i].item())
                if counters[cls] >= samples_per_class:
                    continue

                single = batch_x[i : i + 1].to(device)
                cam, pred_idx, prob = cam_extractor(single, class_idx=cls)

                rgb = _denormalize(single[0])
                heatmap_only = overlay_heatmap(np.zeros_like(rgb), cam)
                overlay = overlay_heatmap(rgb, cam)
                combined = np.concatenate([rgb, heatmap_only, overlay], axis=1)

                fname = (
                    f"{class_names[cls]}_{counters[cls]+1}"
                    f"_pred-{class_names[pred_idx]}_p{prob:.2f}.png"
                )
                save_path = os.path.join(output_dir, fname)
                Image.fromarray(combined).save(save_path)
                saved_paths.append(save_path)
                counters[cls] += 1

            if all(v >= samples_per_class for v in counters.values()):
                break
    finally:
        cam_extractor.remove_hooks()

    print(f"[GradCAM] saved {len(saved_paths)} heatmaps -> {output_dir}")
    return saved_paths


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    from models import build_model
    from utils.dataset import build_dataloaders

    parser = argparse.ArgumentParser(description="Generate GradCAM heatmaps")
    parser.add_argument("--model", default="resnet50",
                        choices=["baseline", "resnet50", "efficientnet"])
    parser.add_argument("--weights", required=True, help="Path to .pth weights")
    parser.add_argument("--data_dir", default="./data/EuroSAT")
    parser.add_argument("--image", default=None,
                        help="If set, generate GradCAM for a single image instead of batch")
    parser.add_argument("--output_dir", default="outputs/gradcam")
    parser.add_argument("--samples_per_class", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    model = build_model(args.model)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)

    if args.image:
        out_path = os.path.join(args.output_dir, "single.png")
        _, idx, prob = generate_gradcam(
            model, args.image, save_path=out_path, device=device
        )
        print(f"Predicted: {EUROSAT_CLASSES[idx]} ({prob*100:.1f}%) -> {out_path}")
    else:
        loaders = build_dataloaders(args.data_dir, batch_size=64, num_workers=0)
        batch_gradcam(
            model,
            loaders["test"],
            output_dir=args.output_dir,
            samples_per_class=args.samples_per_class,
            device=device,
        )
