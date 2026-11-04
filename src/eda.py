"""
Exploratory Data Analysis (EDA) and Visualization Module.
========================================================
Inspects the chest X-ray dataset, plots class distributions, samples images,
evaluates image resolutions and color profiles, and explains class imbalance implications.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.config import config


def compute_dataset_stats(data_dir: Path) -> pd.DataFrame:
    """
    Computes class counts and proportions across train, val, and test splits.
    """
    records = []
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        for cls_name in config.CLASSES:
            cls_dir = split_dir / cls_name
            if cls_dir.exists():
                count = len([f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpeg", ".jpg", ".png"]])
                records.append({
                    "Split": split.capitalize(),
                    "Class": cls_name,
                    "Count": count
                })
    
    df = pd.DataFrame(records)
    print("\n--- Dataset Class Distribution Summary ---")
    print(df.to_string(index=False))
    return df


def plot_class_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> None:
    """
    Plots a multi-split bar chart showing the distribution of NORMAL vs PNEUMONIA cases.
    """
    plt.figure(figsize=(9, 5))
    palette = {"NORMAL": "#2ca02c", "PNEUMONIA": "#d62728"}
    
    ax = sns.barplot(
        data=df,
        x="Split",
        y="Count",
        hue="Class",
        palette=palette,
        edgecolor="black",
        linewidth=1.2
    )
    
    plt.title("Chest X-Ray Class Distribution Across Splits", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Dataset Split", fontsize=12, labelpad=10)
    plt.ylabel("Number of Images", fontsize=12, labelpad=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Condition", frameon=True)
    
    # Add count labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height + 30),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )
            
    plt.ylim(0, df["Count"].max() * 1.15)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] Class distribution plot saved to: {save_path}")
        
    plt.show()
    plt.close()


def inspect_image_dimensions(data_dir: Path, num_samples: int = 50) -> Dict[str, Tuple]:
    """
    Inspects image dimensions, color modes, and aspect ratios from a sample of images.
    """
    train_dir = data_dir / "train"
    image_paths = []
    for cls_name in config.CLASSES:
        cls_dir = train_dir / cls_name
        image_paths.extend(list(cls_dir.glob("*.jpeg"))[:num_samples // 2])
        
    widths = []
    heights = []
    modes = set()
    
    for path in image_paths:
        with Image.open(path) as img:
            widths.append(img.width)
            heights.append(img.height)
            modes.add(img.mode)
            
    summary = {
        "min_width": min(widths),
        "max_width": max(widths),
        "mean_width": int(np.mean(widths)),
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_height": int(np.mean(heights)),
        "color_modes": list(modes)
    }
    
    print("\n--- Image Resolution & Format Diagnostics ---")
    print(f"  - Sampled Images   : {len(image_paths)}")
    print(f"  - Width Range      : {summary['min_width']}px - {summary['max_width']}px (Mean: {summary['mean_width']}px)")
    print(f"  - Height Range     : {summary['min_height']}px - {summary['max_height']}px (Mean: {summary['mean_height']}px)")
    print(f"  - Image Modes Found: {summary['color_modes']} (L = 1-channel Grayscale, RGB = 3-channel)")
    print("  -> Conclusion: High variance in resolution justifies standardized 224x224 RGB resizing.")
    
    return summary


def plot_sample_images(data_dir: Path, n_samples_per_class: int = 5, save_path: Optional[Path] = None) -> None:
    """
    Displays at least 5 sample X-ray images from each class (NORMAL and PNEUMONIA) in a clean grid.
    """
    train_dir = data_dir / "train"
    fig, axes = plt.subplots(2, n_samples_per_class, figsize=(3 * n_samples_per_class, 7))
    
    for row_idx, cls_name in enumerate(config.CLASSES):
        cls_dir = train_dir / cls_name
        img_files = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpeg", ".jpg", ".png"]])
        selected_files = img_files[:n_samples_per_class]
        
        for col_idx, file_path in enumerate(selected_files):
            ax = axes[row_idx, col_idx]
            img = Image.open(file_path)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{cls_name}\n({img.width}x{img.height})", fontsize=10, fontweight="bold")
            ax.axis("off")
            
    plt.suptitle("Sample Pediatric Chest X-Rays (NORMAL vs. PNEUMONIA)", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] Sample images grid saved to: {save_path}")
        
    plt.show()
    plt.close()


def explain_class_imbalance(train_normal: int, train_pneumonia: int) -> str:
    """
    Provides a clear, educational explanation of class imbalance for beginners.
    """
    ratio = train_pneumonia / max(1, train_normal)
    explanation = f"""
================================================================================
CLASS IMBALANCE ANALYSIS & MEDICAL CONTEXT
================================================================================
Observed Counts in Training Data:
  * NORMAL Cases   : {train_normal}
  * PNEUMONIA Cases: {train_pneumonia}
  * Imbalance Ratio: approximately {ratio:.2f}:1 (Pneumonia : Normal)

WHY CLASS IMBALANCE IMPACTS DEEP LEARNING:
1. Skewed Loss Gradients:
   In standard Cross-Entropy Loss, each sample contributes equally to the loss.
   Because ~75% of images are Pneumonia, the model receives 3 times more gradient updates
   encouraging it to predict Pneumonia. A naive model could simply predict 'PNEUMONIA'
   for every single patient and achieve ~75% raw accuracy while being clinically useless!

2. The Asymmetry of Medical Risk:
   - False Negative (Predict Normal when patient has Pneumonia):
     A sick child is sent home untreated; the infection can worsen into severe respiratory
     distress, sepsis, or death. This error is catastrophic.
   - False Positive (Predict Pneumonia when patient is Normal):
     A healthy child is scheduled for further evaluation (e.g. sputum culture, CT scan, or
     second opinion by a senior radiologist). This incurs a small delay or mild cost, but
     does not endanger the patient's life.

OUR MITIGATION STRATEGY:
1. Dynamic Class-Weighted CrossEntropyLoss:
   We penalize misclassifications of the minority class (NORMAL) with a weight of
   weight_normal = n_pneumonia / n_normal (~{ratio:.2f}) while setting weight_pneumonia = 1.0.
2. Threshold Optimization:
   Instead of using a default 0.5 probability cutoff, we calibrate the decision
   threshold on our validation set to guarantee approximately 95% Recall (Sensitivity)
   for detecting Pneumonia cases.
================================================================================
"""
    print(explanation)
    return explanation

