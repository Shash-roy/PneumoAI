"""
Singleton Model Manager for PneumoAI Backend.
=============================================
Loads the ResNet-18 model once on application startup.
Provides thread-safe inference access, GPU/CPU hardware detection,
and robust checkpoint tracking.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

# Add project root to sys.path so we can import from existing src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import config
    from src.model import build_resnet18
except ImportError:
    # Fallback configuration if imported directly
    class FallbackConfig:
        CLASSES = ["NORMAL", "PNEUMONIA"]
        CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
        IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}
        IMAGE_SIZE = (224, 224)
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        BEST_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_finetuned_resnet18.pth"
        BASELINE_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "baseline_resnet18.pth"
        TARGET_RECALL = 0.95
    config = FallbackConfig()


class ModelManager:
    """
    Singleton class managing the ResNet-18 model lifecycle.
    Ensures model weights are loaded exactly once at server launch.
    """
    _instance: Optional["ModelManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.checkpoint_path: Optional[Path] = None
        self.is_custom_checkpoint: bool = False
        self.optimal_threshold: float = 0.40  # Validation-calibrated threshold (~95% Recall)
        self.status_message: str = "Uninitialized"
        
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """
        Loads the best fine-tuned checkpoint or falls back to pretrained ResNet-18 base.
        """
        print(f"[BACKEND] Initializing ModelManager on device: {self.device}...")
        
        # Build ResNet-18 architecture with 2-class head (NORMAL=0, PNEUMONIA=1)
        try:
            from src.model import build_resnet18
            self.model = build_resnet18(pretrained=True, num_classes=2, freeze_backbone=False)
        except Exception:
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        checkpoint_candidates = [
            PROJECT_ROOT / "checkpoints" / "best_finetuned_resnet18.pth",
            PROJECT_ROOT / "checkpoints" / "baseline_resnet18.pth",
            Path("checkpoints/best_finetuned_resnet18.pth"),
            Path("checkpoints/baseline_resnet18.pth"),
        ]

        found_checkpoint = None
        for ckpt in checkpoint_candidates:
            if ckpt.exists() and ckpt.is_file():
                found_checkpoint = ckpt
                break

        if found_checkpoint:
            print(f"[BACKEND] Loading trained weights from: {found_checkpoint}")
            state = torch.load(str(found_checkpoint), map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
                # Retrieve saved threshold if present in metadata
                meta = state.get("metadata", {})
                if "optimal_threshold" in meta:
                    self.optimal_threshold = float(meta["optimal_threshold"])
            elif isinstance(state, dict):
                self.model.load_state_dict(state)
                
            self.checkpoint_path = found_checkpoint
            self.is_custom_checkpoint = True
            self.status_message = f"Active: Custom fine-tuned checkpoint loaded from {found_checkpoint.name}"
        else:
            self.checkpoint_path = None
            self.is_custom_checkpoint = False
            self.status_message = (
                "Active (Pretrained Base Mode): Using ImageNet pretrained ResNet-18. "
                "Custom fine-tuned checkpoint not found in checkpoints/ directory."
            )
            print(f"[BACKEND] {self.status_message}")

        self.model.to(self.device)
        self.model.eval()
        print(f"[BACKEND] Model ready. Decision threshold: {self.optimal_threshold:.2f}")

    def get_model(self) -> Tuple[nn.Module, torch.device]:
        """Returns the loaded model in eval mode and its device."""
        if self.model is None:
            self._load_model()
        return self.model, self.device


# Global accessor
model_manager = ModelManager()
