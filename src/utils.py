"""
Utility Functions for Reproducibility, Device Management, and Visualization.
=============================================================================
Provides essential helper routines used across the training and evaluation pipelines.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Set fixed random seeds across all libraries to ensure 100% reproducible results.
    
    Why this is crucial for beginners:
    Deep learning involves random weight initialization, random data shuffling, and
    stochastic gradient descent. Fixing seeds guarantees that every time you run the code,
    you get the exact same splits, batches, and training trajectories.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure deterministic execution in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed successfully set to: {seed}")


def get_device() -> torch.device:
    """
    Detects hardware accelerator (CUDA GPU or CPU) and logs details.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[HARDWARE] GPU Detected: {gpu_name} ({gpu_memory:.2f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[HARDWARE] No GPU found. Running on CPU (training may be slower).")
    return device


def plot_learning_curves(
    history: Dict[str, List[float]],
    title: str = "Training and Validation Curves",
    save_path: Optional[Path] = None
) -> None:
    """
    Plots Loss and Metric (ROC-AUC / Accuracy) over epochs for training and validation.
    
    Args:
        history: Dictionary containing lists for 'train_loss', 'val_loss',
                 'train_auc', 'val_auc', 'val_acc'.
        title: Title for the overall figure.
        save_path: Optional path to save the plot.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss Subplot
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True)
    
    # 2. Validation ROC-AUC & Accuracy Subplot
    if "val_auc" in history and len(history["val_auc"]) > 0:
        axes[1].plot(epochs, history["val_auc"], "o-", label="Val ROC-AUC", color="#2ca02c", linewidth=2)
    if "val_acc" in history and len(history["val_acc"]) > 0:
        axes[1].plot(epochs, history["val_acc"], "^-.", label="Val Accuracy", color="#d62728", linewidth=2)
    
    axes[1].set_title("Validation Metrics vs. Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    axes[1].set_ylim(0.5, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", frameon=True)
    
    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] Learning curve figure saved to: {save_path}")
    
    plt.show()
    plt.close()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_auc: float,
    path: Path,
    extra_metadata: Optional[dict] = None
) -> None:
    """
    Saves complete model checkpoint including weights, optimizer state, and epoch metrics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc": val_auc,
        "metadata": extra_metadata or {}
    }
    torch.save(payload, str(path))
    print(f"[CHECKPOINT] Model saved to {path} (Val ROC-AUC: {val_auc:.4f})")


def load_checkpoint(
    model: nn.Module,
    path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> dict:
    """
    Loads model weights from saved checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
        
    checkpoint = torch.load(str(path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    print(f"[LOADED] Checkpoint loaded from {path} (Epoch: {checkpoint.get('epoch', 'N/A')}, Val AUC: {checkpoint.get('val_auc', 'N/A')})")
    return checkpoint

