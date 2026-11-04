"""
Training and Optimization Pipeline.
===================================
Implements:
1. One-epoch training and validation step functions with progress tracking.
2. Dynamic class-weight calculation for handling class imbalance.
3. Baseline feature-extraction training loop.
4. End-to-end Fine-tuning loop with Early Stopping based on validation ROC-AUC.
5. Checkpoint management.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import config
from src.utils import plot_learning_curves, save_checkpoint


def compute_class_weights(
    n_normal: int,
    n_pneumonia: int,
    device: torch.device
) -> torch.Tensor:
    """
    Computes dynamic class weights according to the project specifications:
        class_weights = torch.tensor([n_pneumonia / n_normal, 1.0], device=device)
        
    This ensures that errors made on the minority class (NORMAL) incur a larger loss penalty,
    preventing the model from ignoring the normal class due to pneumonia dominance.
    """
    weight_normal = float(n_pneumonia) / float(max(1, n_normal))
    weight_pneumonia = 1.0
    weights = torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32, device=device)
    print(f"[CLASS WEIGHTS] Computed: NORMAL={weight_normal:.4f}, PNEUMONIA={weight_pneumonia:.4f}")
    return weights


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Executes one full training epoch over the dataloader.
    
    Returns:
        (epoch_loss, epoch_accuracy, epoch_roc_auc)
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    all_preds = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        # Softmax probabilities for ROC-AUC
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    total_samples = len(dataloader.dataset)
    epoch_loss = running_loss / total_samples
    epoch_acc = accuracy_score(all_targets, all_preds)
    
    # In rare single-batch edge cases, roc_auc can fail if only one class exists
    try:
        epoch_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_acc, epoch_auc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluates the model over the validation or test dataloader.
    
    Returns:
        (eval_loss, eval_accuracy, eval_roc_auc, all_targets, all_probs)
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    all_preds = []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    total_samples = len(dataloader.dataset)
    eval_loss = running_loss / total_samples
    eval_acc = accuracy_score(all_targets, all_preds)
    
    try:
        eval_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        eval_auc = 0.5

    return eval_loss, eval_acc, eval_auc, np.array(all_targets), np.array(all_probs)


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = config.BASELINE_EPOCHS,
    lr: float = config.BASELINE_LR,
    weight_decay: float = config.BASELINE_WEIGHT_DECAY,
    checkpoint_path: Path = config.BASELINE_MODEL_PATH
) -> Dict[str, List[float]]:
    """
    Step 1: Train the Baseline Model (Feature Extractor Frozen).
    Only the final classification head is trained using CrossEntropyLoss and AdamW.
    """
    print("\n" + "=" * 80)
    print("STAGE 1: TRAINING BASELINE MODEL (CONVOLUTIONAL BACKBONE FROZEN)")
    print("=" * 80)
    
    # Only optimize parameters of the head (requires_grad == True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_auc": [], "val_auc": []
    }

    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_auc, _, _ = evaluate_epoch(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val ROC-AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_auc=val_auc,
                path=checkpoint_path,
                extra_metadata={"stage": "baseline", "val_acc": val_acc}
            )

    plot_learning_curves(
        history,
        title="Baseline ResNet-18 (Head Only) Training Curves",
        save_path=config.OUTPUT_DIR / "baseline_learning_curves.png"
    )
    return history


def train_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int = config.FINETUNE_EPOCHS,
    lr: float = config.FINETUNE_LR,
    weight_decay: float = config.FINETUNE_WEIGHT_DECAY,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    checkpoint_path: Path = config.BEST_MODEL_PATH
) -> Dict[str, List[float]]:
    """
    Step 2: Fine-Tune the Complete ResNet-18 Network.
    - All layers unfrozen.
    - Gentle learning rate (1e-4) to avoid disrupting pretrained feature filters.
    - Class-weighted CrossEntropyLoss to address class imbalance.
    - Early stopping tracking validation ROC-AUC.
    - NEVER evaluate on test set during model selection!
    """
    print("\n" + "=" * 80)
    print("STAGE 2: FULL FINE-TUNING OF RESNET-18 (ALL LAYERS UNFROZEN + CLASS WEIGHTS)")
    print("=" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_auc": [], "val_auc": []
    }

    best_val_auc = -1.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_auc, _, _ = evaluate_epoch(
            model, val_loader, criterion, device
        )
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)

        is_best = val_auc > best_val_auc
        marker = " [★ BEST]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val ROC-AUC: {val_auc:.4f}{marker}"
        )

        if is_best:
            best_val_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_auc=val_auc,
                path=checkpoint_path,
                extra_metadata={"stage": "fine_tuned", "val_acc": val_acc}
            )
        else:
            patience_counter += 1
            print(f"  -> EarlyStopping counter: {patience_counter} of {patience}")
            if patience_counter >= patience:
                print(f"[EARLY STOPPING] Validation ROC-AUC did not improve for {patience} consecutive epochs. Stopping.")
                break

    plot_learning_curves(
        history,
        title="Fine-Tuned ResNet-18 (Full Backbone) Training Curves",
        save_path=config.OUTPUT_DIR / "finetuned_learning_curves.png"
    )
    return history

