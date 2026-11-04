"""
CLI Script: Model Evaluation, Threshold Optimization & Grad-CAM Analysis.
========================================================================
Executes:
1. Loads best fine-tuned ResNet-18 model checkpoint.
2. Calibrates optimal decision threshold on the VALIDATION set to achieve ~95% Recall.
3. Evaluates both Default (0.50) and Optimal thresholds on the untouched TEST set.
4. Generates:
   - Confusion Matrices
   - ROC & Precision-Recall Curves
   - Diagnostic Performance Comparison Table
   - Grad-CAM Saliency Grids
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.config import config
from src.dataset import (
    create_stratified_train_val_split,
    find_dataset_dir,
    get_dataloaders,
)
from src.evaluate import (
    calculate_clinical_metrics,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    print_comparison_table,
)
from src.explainability import plot_explainability_grid
from src.model import build_resnet18
from src.train import evaluate_epoch
from src.utils import get_device, load_checkpoint, set_seed


def main():
    print("=" * 80)
    print("STEP 3: COMPREHENSIVE TEST EVALUATION & CLINICAL EXPLAINABILITY")
    print("=" * 80)

    set_seed(config.SEED)
    device = get_device()

    # 1. Load Data
    data_dir = find_dataset_dir()
    train_ds, val_ds, test_ds, _ = create_stratified_train_val_split(data_dir)
    _, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds)

    # 2. Build Model & Load Best Weights
    model = build_resnet18(pretrained=False, num_classes=config.NUM_CLASSES, freeze_backbone=False).to(device)
    
    checkpoint_to_load = config.BEST_MODEL_PATH
    if not checkpoint_to_load.exists():
        print(f"[NOTICE] Best checkpoint {checkpoint_to_load} not found. Checking baseline checkpoint...")
        checkpoint_to_load = config.BASELINE_MODEL_PATH
        
    if checkpoint_to_load.exists():
        load_checkpoint(model, checkpoint_to_load, device=device)
    else:
        print("[WARNING] No saved checkpoint found. Running evaluation on default initialization.")

    criterion = nn.CrossEntropyLoss()

    # 3. Optimize Threshold on Validation Set
    print("\n--- Calibrating Threshold on Validation Set ---")
    _, _, _, y_val_true, y_val_prob = evaluate_epoch(model, val_loader, criterion, device)
    optimal_threshold, val_opt_metrics = find_optimal_threshold(
        y_val_true, y_val_prob, target_recall=config.TARGET_RECALL
    )

    # 4. Evaluate on Untouched Test Set
    print("\n--- Evaluating on Untouched Kaggle Test Set ---")
    _, _, test_auc, y_test_true, y_test_prob = evaluate_epoch(model, test_loader, criterion, device)

    # Metrics at Default Threshold (0.50)
    test_default_metrics = calculate_clinical_metrics(y_test_true, y_test_prob, threshold=0.50)
    # Metrics at Optimal Threshold
    test_optimal_metrics = calculate_clinical_metrics(y_test_true, y_test_prob, threshold=optimal_threshold)

    # Print Clinical Comparison Table
    print_comparison_table(test_default_metrics, test_optimal_metrics)

    # 5. Visualizations
    # Confusion Matrix: Default
    plot_confusion_matrix(
        y_test_true,
        (y_test_prob >= 0.50).astype(int),
        title=f"Test Confusion Matrix (Default Threshold: 0.50)\nRecall: {test_default_metrics['recall']*100:.1f}%",
        save_path=config.OUTPUT_DIR / "confusion_matrix_default_0.50.png"
    )

    # Confusion Matrix: Optimal Threshold
    plot_confusion_matrix(
        y_test_true,
        (y_test_prob >= optimal_threshold).astype(int),
        title=f"Test Confusion Matrix (Optimal Threshold: {optimal_threshold:.2f})\nRecall: {test_optimal_metrics['recall']*100:.1f}%",
        save_path=config.OUTPUT_DIR / f"confusion_matrix_optimal_{optimal_threshold:.2f}.png"
    )

    # ROC and PR Curves
    plot_roc_and_pr_curves(
        y_test_true,
        y_test_prob,
        optimal_threshold=optimal_threshold,
        save_path=config.OUTPUT_DIR / "roc_and_pr_curves.png"
    )

    # 6. Grad-CAM Explainability Grid on Representative Test Images
    test_dir = data_dir / "test"
    sample_images = []
    
    # Grab 2 Normal and 2 Pneumonia images
    normal_samples = list((test_dir / "NORMAL").glob("*.jpeg"))[:2]
    pneumonia_samples = list((test_dir / "PNEUMONIA").glob("*.jpeg"))[:2]
    sample_images.extend(normal_samples + pneumonia_samples)

    if sample_images:
        print("\n--- Generating Grad-CAM Diagnostic Saliency Heatmaps ---")
        plot_explainability_grid(
            model=model,
            sample_paths=sample_images,
            device=device,
            save_path=config.OUTPUT_DIR / "gradcam_test_samples.png"
        )

    print("\n[SUCCESS] Full Evaluation & Explainability Analysis completed!")
    print(f"Figures saved in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

