"""
CLI Script: Complete Training Pipeline (Baseline + Fine-Tuning).
===============================================================
Orchestrates:
1. Environment setup and reproducible seed initialization.
2. Dataset path detection and stratified 85:15 split creation.
3. Phase 1: Baseline Training (ResNet-18 Backbone Frozen).
4. Phase 2: Full Fine-Tuning (All Layers Unfrozen + Dynamic Class Weights).
5. Validation ROC-AUC model selection and checkpoint preservation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.dataset import (
    create_stratified_train_val_split,
    find_dataset_dir,
    get_dataloaders,
    verify_dataset_structure,
)
from src.model import build_resnet18, get_parameter_summary, unfreeze_all_layers
from src.train import compute_class_weights, train_baseline, train_finetune
from src.utils import get_device, set_seed


def main():
    print("=" * 80)
    print("PNEUMONIA DETECTION: END-TO-END MODEL TRAINING PIPELINE")
    print("=" * 80)

    # 1. Reproducibility & Device
    set_seed(config.SEED)
    device = get_device()

    # 2. Dataset Discovery & Stratified Split
    data_dir = find_dataset_dir()
    verify_dataset_structure(data_dir)
    
    train_ds, val_ds, test_ds, split_info = create_stratified_train_val_split(data_dir)
    train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds)

    # 3. Dynamic Class Weights for Imbalance
    class_weights = compute_class_weights(
        n_normal=split_info["train_normal"],
        n_pneumonia=split_info["train_pneumonia"],
        device=device
    )

    # 4. Phase 1: Baseline Training (Feature Extractor Frozen)
    print("\n--- Constructing Baseline Model (ResNet-18) ---")
    baseline_model = build_resnet18(
        pretrained=True,
        num_classes=config.NUM_CLASSES,
        freeze_backbone=True
    ).to(device)
    get_parameter_summary(baseline_model)

    baseline_history = train_baseline(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=config.BASELINE_EPOCHS,
        lr=config.BASELINE_LR
    )

    # 5. Phase 2: Fine-Tuning (All Layers Unfrozen)
    print("\n--- Preparing Model for Full Fine-Tuning ---")
    unfreeze_all_layers(baseline_model)
    get_parameter_summary(baseline_model)

    finetune_history = train_finetune(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
        epochs=config.FINETUNE_EPOCHS,
        lr=config.FINETUNE_LR,
        patience=config.EARLY_STOPPING_PATIENCE
    )

    print("\n" + "=" * 80)
    print("[TRAINING COMPLETE] Model checkpoints and performance plots saved successfully!")
    print(f"  - Baseline Checkpoint  : {config.BASELINE_MODEL_PATH}")
    print(f"  - Best Fine-Tuned Model: {config.BEST_MODEL_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()

