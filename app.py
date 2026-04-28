"""
app.py
======
课程要求 / Course Requirements:
    - Web Demo ✅
    - Transfer Learning / Deep Learning / PyTorch ✅

Flask 后端服务. 提供以下 endpoints:

    GET  /              -> web/index.html
    GET  /<asset>       -> 静态资源 (style.css, main.js)
    POST /predict       -> 接收图片, 返回 JSON: class / confidence /
                           all_probs / gradcam_img(base64 PNG)
    GET  /results       -> outputs/metrics.json 的内容 (供前端模型对比表使用)
    GET  /health        -> 健康检查

加载 ResNet-50 最优权重 (outputs/resnet50_best.pth);
若权重不存在, /predict 会返回友好提示信息.
"""

from __future__ import annotations

import base64
import io
import json
import os
import traceback
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError

from gradcam import GradCAM, _denormalize, _get_default_target_layer, overlay_heatmap
from models import build_model
from utils.dataset import (
    EUROSAT_CLASSES,
    EUROSAT_EMOJI,
    get_inference_transform,
)
from utils.trainer import get_device

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DEFAULT_WEIGHTS = os.path.join(OUTPUT_DIR, "resnet50_best.pth")
DEFAULT_MODEL_NAME = "resnet50"
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

DEVICE = get_device()
MODEL: Optional[torch.nn.Module] = None
TRANSFORM = get_inference_transform()


# -----------------------------------------------------------------------------
# Lazy model loader
# -----------------------------------------------------------------------------
def _load_model() -> Optional[torch.nn.Module]:
    """Lazy-load the ResNet-50 best checkpoint. Returns None if missing."""
    global MODEL
    if MODEL is not None:
        return MODEL

    if not os.path.exists(DEFAULT_WEIGHTS):
        print(
            f"[App] ⚠ Weights not found: {DEFAULT_WEIGHTS}. "
            f"Run 'python train.py --model resnet50' first."
        )
        return None

    try:
        model = build_model(DEFAULT_MODEL_NAME, num_classes=10, finetune_mode="partial")
        state = torch.load(DEFAULT_WEIGHTS, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        MODEL = model
        print(f"[App] ✓ Loaded {DEFAULT_MODEL_NAME} weights from {DEFAULT_WEIGHTS}")
        return MODEL
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[App] Failed to load model: {exc}")
        traceback.print_exc()
        return None


def _allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXT


def _np_to_base64_png(rgb_array: np.ndarray) -> str:
    """(H,W,3) uint8 -> base64 PNG string."""
    img = Image.fromarray(rgb_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# -----------------------------------------------------------------------------
# Routes — static
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def static_assets(filename: str):
    """Serve files placed in /web (style.css, main.js, etc.)."""
    full = os.path.join(WEB_DIR, filename)
    if os.path.isfile(full):
        return send_from_directory(WEB_DIR, filename)
    return jsonify({"error": "not found"}), 404


# -----------------------------------------------------------------------------
# Routes — API
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "model_loaded": MODEL is not None,
            "weights_exists": os.path.exists(DEFAULT_WEIGHTS),
        }
    )


@app.route("/results", methods=["GET"])
def results():
    """Return the cumulative metrics.json so the frontend can render the table."""
    json_path = os.path.join(OUTPUT_DIR, "metrics.json")
    if not os.path.exists(json_path):
        return jsonify(
            {
                "available": False,
                "message": "No metrics.json yet. Run 'python train.py' first.",
            }
        )
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return jsonify({"available": False, "error": str(exc)}), 500

    summary = []
    for model_name, m in data.items():
        summary.append(
            {
                "model": model_name,
                "accuracy": m.get("overall_accuracy", 0.0),
                "macro_f1": m.get("macro_f1", 0.0),
            }
        )
    return jsonify({"available": True, "summary": summary, "raw": data})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict the EuroSAT class of an uploaded image + return GradCAM."""
    model = _load_model()
    if model is None:
        return (
            jsonify(
                {
                    "error": (
                        "Model weights not found. Please train ResNet-50 first: "
                        "`python train.py --model resnet50`."
                    )
                }
            ),
            503,
        )

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded under field 'image'."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not _allowed_file(file.filename):
        return (
            jsonify(
                {"error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXT)}"}
            ),
            400,
        )

    # 解码图像
    try:
        pil = Image.open(file.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "Cannot decode the uploaded image."}), 400

    try:
        x = TRANSFORM(pil).unsqueeze(0).to(DEVICE)

        # 1) 前向 + softmax 概率
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            confidence = float(probs[pred_idx].item())

        all_probs = {
            EUROSAT_CLASSES[i]: float(probs[i].item())
            for i in range(len(EUROSAT_CLASSES))
        }

        # 2) GradCAM (新建一份输入张量, 因为前一段是 no_grad)
        x_grad = TRANSFORM(pil).unsqueeze(0).to(DEVICE)
        target_layer = _get_default_target_layer(model)
        with GradCAM(model, target_layer) as cam_extractor:
            cam, _, _ = cam_extractor(x_grad, class_idx=pred_idx)
        rgb = _denormalize(x[0])
        overlay = overlay_heatmap(rgb, cam)
        gradcam_b64 = _np_to_base64_png(overlay)

        return jsonify(
            {
                "class": EUROSAT_CLASSES[pred_idx],
                "class_emoji": EUROSAT_EMOJI.get(EUROSAT_CLASSES[pred_idx], ""),
                "confidence": confidence,
                "all_probs": all_probs,
                "gradcam_img": gradcam_b64,
            }
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    _load_model()  # 提前加载, 提高首次请求响应速度
    print(f"[App] Open http://127.0.0.1:5000 in your browser.")
    app.run(host="0.0.0.0", port=5000, debug=False)
