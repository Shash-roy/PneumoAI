"""
Configuration Module for Pneumonia Detection Project.
=====================================================
Centralizes all hyperparameters, image normalization statistics, file paths,
and reproducibility seeds. Keeping configurations in one place makes experiments
reproducible and easy to tune without altering core logic.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

try:
    import torch
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEFAULT_DEVICE = "cpu"


@dataclass
class Config:
    # --------------------------------------------------------------------------
    # Reproducibility & Hardware
    # --------------------------------------------------------------------------
    SEED: int = 42
    DEVICE: str = DEFAULT_DEVICE
    NUM_WORKERS: int = 2

    # --------------------------------------------------------------------------
    # Dataset & Class Labels
    # --------------------------------------------------------------------------
    # Kaggle dataset identifier: paultimothymooney/chest-xray-pneumonia
    KAGGLE_DATASET_ID: str = "paultimothymooney/chest-xray-pneumonia"
    
    # Class mappings (PyTorch ImageFolder assigns labels alphabetically: NORMAL=0, PNEUMONIA=1)
    CLASSES: List[str] = field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])
    CLASS_TO_IDX: dict = field(default_factory=lambda: {"NORMAL": 0, "PNEUMONIA": 1})
    IDX_TO_CLASS: dict = field(default_factory=lambda: {0: "NORMAL", 1: "PNEUMONIA"})
    NUM_CLASSES: int = 2

    # --------------------------------------------------------------------------
    # Image Preprocessing & Normalization
    # --------------------------------------------------------------------------
    # ResNet-18 expects 224x224 RGB images normalized with ImageNet statistics
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    IMAGENET_MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    IMAGENET_STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # --------------------------------------------------------------------------
    # Data Split Parameters
    # --------------------------------------------------------------------------
    # Fix the original 16-image validation folder by creating a stratified 85:15 split
    # from the combined original (train + val) images. Test set is left untouched!
    TRAIN_VAL_SPLIT_RATIO: float = 0.85
    VAL_RATIO: float = 0.15

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE: int = 32
    
    # Baseline Model (Freeze feature extractor, train only final FC layer)
    BASELINE_EPOCHS: int = 5
    BASELINE_LR: float = 1e-3
    BASELINE_WEIGHT_DECAY: float = 1e-4

    # Fine-Tuning Model (Unfreeze all layers, lower LR, class-weighted loss)
    FINETUNE_EPOCHS: int = 15
    FINETUNE_LR: float = 1e-4
    FINETUNE_WEIGHT_DECAY: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 4  # Stop if validation ROC-AUC doesn't improve for 4 epochs

    # Threshold Optimization
    TARGET_RECALL: float = 0.95  # Prioritize detecting pneumonia to minimize false negatives

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
    
    # Model artifact paths
    BASELINE_MODEL_PATH: Path = CHECKPOINT_DIR / "baseline_resnet18.pth"
    BEST_MODEL_PATH: Path = CHECKPOINT_DIR / "best_finetuned_resnet18.pth"

    def __post_init__(self):
        """Ensure directories exist."""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# Default global configuration instance
config = Config()
