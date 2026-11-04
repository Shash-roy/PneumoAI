"""
Evaluation, Threshold Optimization, and Diagnostic Metrics Module.
==================================================================
Implements:
1. Full diagnostic metric evaluation (Accuracy, Sensitivity/Recall, Specificity, Precision, F1, ROC-AUC).
2. Confusion Matrix visualization with raw counts and normalized rates.
3. ROC Curve and Precision-Recall Curve plotting.
4. Threshold optimization on the validation set targeting ~95% Recall for Pneumonia detection.
5. Comparative evaluation of Default (0.50) vs. Optimal threshold on the unseen test set.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import config


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, float]:
    """
    Computes standard and clinical metrics for a given decision threshold.
    
    Terminology:
    - Positive Class (1) = PNEUMONIA
    - Negative Class (0) = NORMAL
    - Sensitivity / Recall = TP / (TP + FN) -> Percentage of sick patients correctly identified
    - Specificity = TN / (TN + FP) -> Percentage of healthy patients correctly identified
    - Precision / PPV = TP / (TP + FP) -> When model flags pneumonia, how often is it right?
    - F1-Score = Harmonic mean of Precision and Recall
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


def find_optimal_threshold(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    target_recall: float = config.TARGET_RECALL
) -> Tuple[float, Dict[str, float]]:
    """
    Sweeps thresholds on the VALIDATION set to find the optimal decision cutoff
    that achieves approximately target_recall (default: 0.95) while maximizing F1 / precision.
    
    Why calibrate on the VALIDATION set?
    Never tune decision thresholds on the TEST set! The test set must act as an independent,
    unbiased benchmark of real-world generalization. Tuning on validation prevents data leakage.
    """
    thresholds = np.linspace(0.01, 0.99, 500)
    best_threshold = 0.50
    best_candidate_metrics = None
    min_recall_diff = float("inf")
    highest_f1_meeting_recall = -1.0

    # Step 1: Look for thresholds achieving >= target_recall with highest F1
    candidates_meeting_target = []
    for t in thresholds:
        m = calculate_clinical_metrics(y_val_true, y_val_prob, threshold=t)
        if m["recall"] >= target_recall:
            candidates_meeting_target.append((t, m))

    if candidates_meeting_target:
        # Choose the one with the highest F1 score among those that satisfy the recall threshold
        best_threshold, best_candidate_metrics = max(candidates_meeting_target, key=lambda x: x[1]["f1_score"])
    else:
        # If 95% recall is not strictly reachable, select threshold yielding closest recall
        for t in thresholds:
            m = calculate_clinical_metrics(y_val_true, y_val_prob, threshold=t)
            diff = abs(m["recall"] - target_recall)
            if diff < min_recall_diff:
                min_recall_diff = diff
                best_threshold = t
                best_candidate_metrics = m

    print("\n" + "=" * 80)
    print(f"THRESHOLD OPTIMIZATION RESULTS (CALIBRATED ON VALIDATION SET)")
    print("=" * 80)
    print(f"  - Target Recall     : {target_recall * 100:.1f}%")
    print(f"  - Optimal Threshold : {best_threshold:.4f}")
    print(f"  - Val Recall        : {best_candidate_metrics['recall'] * 100:.2f}%")
    print(f"  - Val Specificity   : {best_candidate_metrics['specificity'] * 100:.2f}%")
    print(f"  - Val Precision     : {best_candidate_metrics['precision'] * 100:.2f}%")
    print(f"  - Val F1-Score      : {best_candidate_metrics['f1_score'] * 100:.2f}%")
    print("=" * 80)

    return float(best_threshold), best_candidate_metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None
) -> None:
    """
    Plots an annotated Confusion Matrix displaying raw sample counts and percentages.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = [f"{c}\n(True)" for c in config.CLASSES]
    
    # Annotations with counts and row percentages
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=config.CLASSES,
        yticklabels=config.CLASSES,
        annot_kws={"size": 13, "weight": "bold"},
        linewidths=1,
        linecolor="black"
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=12, labelpad=10)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] Confusion matrix saved to: {save_path}")

    plt.show()
    plt.close()


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_threshold: float,
    save_path: Optional[Path] = None
) -> None:
    """
    Plots side-by-side ROC Curve and Precision-Recall Curve, marking the optimal threshold operating point.
    """
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    roc_score = auc(fpr, tpr)

    precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob)
    pr_score = auc(recall, precision)

    # Find operating points for optimal threshold
    opt_idx_roc = np.argmin(np.abs(roc_thresh - optimal_threshold))
    opt_idx_pr = np.argmin(np.abs(pr_thresh - optimal_threshold)) if len(pr_thresh) > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ResNet-18 (AUC = {roc_score:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    axes[0].scatter(
        [fpr[opt_idx_roc]], [tpr[opt_idx_roc]],
        color="red", s=100, zorder=5,
        label=f"Opt Thresh ({optimal_threshold:.2f}): Recall={tpr[opt_idx_roc]*100:.1f}%"
    )
    axes[0].set_xlim([-0.02, 1.02])
    axes[0].set_ylim([-0.02, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Precision-Recall Curve
    axes[1].plot(recall, precision, color="#2ca02c", lw=2.5, label=f"PR Curve (AUC = {pr_score:.4f})")
    if opt_idx_pr < len(recall) and opt_idx_pr < len(precision):
        axes[1].scatter(
            [recall[opt_idx_pr]], [precision[opt_idx_pr]],
            color="red", s=100, zorder=5,
            label=f"Opt Thresh ({optimal_threshold:.2f}): Prec={precision[opt_idx_pr]*100:.1f}%"
        )
    axes[1].set_xlim([-0.02, 1.02])
    axes[1].set_ylim([-0.02, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower left", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] ROC and PR curves saved to: {save_path}")

    plt.show()
    plt.close()


def print_comparison_table(
    default_metrics: Dict[str, float],
    optimal_metrics: Dict[str, float]
) -> None:
    """
    Displays a comparison table between default (0.50) and optimal (~95% recall) thresholds.
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC PERFORMANCE COMPARISON ON UNTOUCHED TEST SET")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Default Thresh (0.50)':<22} | {'Optimal Thresh (' + f'{optimal_metrics[\"threshold\"]:.3f})':<22}")
    print("-" * 80)
    print(f"{'Recall (Sensitivity)':<25} | {default_metrics['recall']*100:>20.2f}% | {optimal_metrics['recall']*100:>20.2f}%")
    print(f"{'Specificity':<25} | {default_metrics['specificity']*100:>20.2f}% | {optimal_metrics['specificity']*100:>20.2f}%")
    print(f"{'Precision (PPV)':<25} | {default_metrics['precision']*100:>20.2f}% | {optimal_metrics['precision']*100:>20.2f}%")
    print(f"{'Accuracy':<25} | {default_metrics['accuracy']*100:>20.2f}% | {optimal_metrics['accuracy']*100:>20.2f}%")
    print(f"{'F1-Score':<25} | {default_metrics['f1_score']*100:>20.2f}% | {optimal_metrics['f1_score']*100:>20.2f}%")
    print(f"{'ROC-AUC':<25} | {default_metrics['roc_auc']:>21.4f} | {optimal_metrics['roc_auc']:>21.4f}")
    print("-" * 80)
    print(f"{'True Positives (TP)':<25} | {default_metrics['tp']:>21} | {optimal_metrics['tp']:>21}")
    print(f"{'False Negatives (FN)':<25} | {default_metrics['fn']:>21} | {optimal_metrics['fn']:>21}  <-- Critically reduced!")
    print(f"{'True Negatives (TN)':<25} | {default_metrics['tn']:>21} | {optimal_metrics['tn']:>21}")
    print(f"{'False Positives (FP)':<25} | {default_metrics['fp']:>21} | {optimal_metrics['fp']:>21}")
    print("=" * 80)
