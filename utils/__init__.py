"""
Utils package for EuroSAT Land Use Classification project.
Course requirements covered: PyTorch / Deep Learning / Measure Performance
"""

from .dataset import build_dataloaders, EUROSAT_CLASSES, EUROSAT_MEAN, EUROSAT_STD
from .trainer import train_model, get_device
from .evaluator import evaluate_model
from .visualizer import plot_training_curves, plot_model_comparison

__all__ = [
    "build_dataloaders",
    "EUROSAT_CLASSES",
    "EUROSAT_MEAN",
    "EUROSAT_STD",
    "train_model",
    "get_device",
    "evaluate_model",
    "plot_training_curves",
    "plot_model_comparison",
]
