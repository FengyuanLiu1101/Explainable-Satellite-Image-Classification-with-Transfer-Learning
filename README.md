# EuroSAT Land Use Classification — Deep Learning Term Project

A **Transfer Learning** project that classifies 64×64 RGB satellite imagery
from the [EuroSAT dataset](https://github.com/phelber/EuroSAT) into 10
land-use categories. Three architectures are compared:

| Model | Strategy | Purpose |
|-------|----------|---------|
| **BaselineCNN** | Trained from scratch | Control group — quantify the value of transfer learning |
| **ResNet-50** | ImageNet pretrained, partial fine-tune (`layer3` + `layer4` + head) | Primary transfer-learning model |
| **EfficientNet-B0** | ImageNet pretrained, partial fine-tune | Lightweight transfer-learning alternative |

The project covers all four required topics for the course:

- ✅ **Transfer Learning** (ImageNet pretrained ResNet-50 / EfficientNet-B0)
- ✅ **Deep Learning** (custom CNN + deep transfer models)
- ✅ **PyTorch** (training, fine-tuning, hooks, GradCAM)
- ✅ **Measure Performance** (Accuracy, Macro F1, per-class accuracy,
  classification report, confusion matrix, training curves, GradCAM)
- ✅ **Web Demo** (Flask backend + dark-themed responsive frontend)

---

## 📦 Dataset

EuroSAT contains **27,000** RGB images (64×64) across 10 classes:

```
AnnualCrop  Forest  HerbaceousVegetation  Highway  Industrial
Pasture     PermanentCrop  Residential   River    SeaLake
```

Download the RGB version from the official repo and unzip into `./data/`:

```
data/EuroSAT/
├── AnnualCrop/        # 3,000 .jpg
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

Direct link: <https://github.com/phelber/EuroSAT>

The pipeline performs a **70 / 15 / 15** train / val / test split with
`seed=42` (see `utils/dataset.py`).

---

## 🛠 Installation

```bash
# 1. (optional) create a clean virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt
```

> **CUDA users**: install a CUDA-enabled PyTorch build from
> <https://pytorch.org/get-started/locally/> *before* `pip install -r requirements.txt`
> if you want GPU acceleration.

---

## 🚀 Training

Train all three models sequentially and produce the comparison plot:

```bash
python train.py --model all --epochs 20 --data_dir ./data/EuroSAT
```

Train a single model:

```bash
# Control baseline (from scratch)
python train.py --model baseline --epochs 20

# ResNet-50 with the default partial fine-tune strategy
python train.py --model resnet50 --epochs 20 --mode partial

# Try other fine-tuning regimes
python train.py --model resnet50 --mode full_freeze
python train.py --model resnet50 --mode full_finetune

# EfficientNet-B0
python train.py --model efficientnet --epochs 20 --mode partial
```

Outputs are written to `outputs/`:

- `<model>_best.pth`              — best-val-accuracy checkpoint
- `<model>_curves.png`            — loss / accuracy curves
- `<model>_confusion_matrix.png`  — normalized confusion matrix
- `model_comparison.png`          — bar chart (when `--model all`)
- `metrics.json`                  — cumulative metrics for all models
- `train.log`                     — full training log

---

## 📊 Evaluation

Re-evaluate a saved checkpoint at any time:

```bash
python evaluate.py --model resnet50 \
                   --weights outputs/resnet50_best.pth \
                   --data_dir ./data/EuroSAT
```

Generate GradCAM heatmaps (3 samples per class on the test set):

```bash
python gradcam.py --model resnet50 \
                  --weights outputs/resnet50_best.pth \
                  --data_dir ./data/EuroSAT \
                  --output_dir outputs/gradcam
```

Single-image GradCAM:

```bash
python gradcam.py --model resnet50 \
                  --weights outputs/resnet50_best.pth \
                  --image path/to/your_image.jpg
```

---

## 🌐 Web Demo

```bash
python app.py
# -> http://127.0.0.1:5000
```

Features:

- Drag-and-drop or click-to-upload satellite images
- Real-time prediction (top-1 class + emoji + confidence bar)
- Animated bar chart of all 10 class probabilities
- Side-by-side **input image** and **GradCAM overlay**
- Live model-comparison table sourced from `outputs/metrics.json`
- Friendly error handling (server / model / file-format problems)

> If `outputs/resnet50_best.pth` is missing the API returns a 503 with a
> clear hint to run `python train.py --model resnet50` first.

---

## 🗂 Project Structure

```
eurosat_project/
├── data/                        # EuroSAT dataset (download manually)
├── models/
│   ├── __init__.py              # build_model factory
│   ├── baseline_cnn.py          # 4-layer CNN (control group)
│   ├── resnet50.py              # ResNet-50 transfer learning
│   └── efficientnet.py          # EfficientNet-B0 transfer learning
├── utils/
│   ├── __init__.py
│   ├── dataset.py               # ImageFolder + 70/15/15 split + augmentation
│   ├── trainer.py               # AdamW + Cosine LR + best-checkpoint saver
│   ├── evaluator.py             # Accuracy / F1 / confusion matrix / metrics.json
│   └── visualizer.py            # Training curves + model comparison plot
├── web/
│   ├── index.html               # Single-page Web Demo
│   ├── style.css                # Dark theme, #00d4aa accent
│   └── main.js                  # Drag-drop + fetch + animated charts
├── outputs/                     # Created at runtime (weights, plots, logs)
├── train.py                     # Main training entry-point
├── evaluate.py                  # Standalone evaluation entry-point
├── gradcam.py                   # Manual GradCAM (no third-party libs)
├── app.py                       # Flask backend
├── requirements.txt
└── README.md
```

---

## 🧪 Experimental Results *(fill in after running)*

| Model            | Test Accuracy | Macro F1 | # Params (trainable) | Notes                  |
|------------------|--------------:|---------:|---------------------:|------------------------|
| BaselineCNN      |          ___% |    ___%  |              ___ M    | from scratch          |
| ResNet-50        |          ___% |    ___%  |              ___ M    | partial fine-tune     |
| EfficientNet-B0  |          ___% |    ___%  |              ___ M    | partial fine-tune     |

**Per-class accuracy (best model)** — see `outputs/metrics.json`.

**GradCAM observations** — paste qualitative observations of `outputs/gradcam/`
images here (e.g., the model focuses on river edges for `River`,
on dense canopy texture for `Forest`, etc.).

---

## 🔁 Reproducibility

All random seeds are set to `42` in `utils.dataset.set_seed()`. The same
seed is propagated to the train / val / test split and the PyTorch
generators, so re-running `python train.py` should yield deterministic splits
(the optimization itself remains subject to non-determinism on GPU
unless `torch.backends.cudnn.deterministic` is enabled).

---

## 📚 References

- Helber, P. et al. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for
  Land Use and Land Cover Classification.* IEEE JSTARS, 2019.
- He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Tan, M. & Le, Q. *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019.
- Selvaraju, R. R. et al. *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization.* ICCV 2017.
