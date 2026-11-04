"""
CLI Script: Run Exploratory Data Analysis (EDA).
===============================================
Executes:
1. Intelligent dataset path detection.
2. Dataset structure verification and image counts.
3. Class distribution bar charts across train, val, and test.
4. Resolution, format, and aspect ratio checks.
5. Visual grid of sample normal and pneumonia X-rays.
6. Educational summary of class imbalance and medical implications.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.dataset import find_dataset_dir, verify_dataset_structure
from src.eda import (
    compute_dataset_stats,
    explain_class_imbalance,
    inspect_image_dimensions,
    plot_class_distribution,
    plot_sample_images,
)


def main():
    print("=" * 80)
    print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    # 1. Locate dataset
    dataset_dir = find_dataset_dir()
    print(f"[DATASET ROOT] {dataset_dir}")

    # 2. Verify structure
    counts = verify_dataset_structure(dataset_dir)

    # 3. Compute stats dataframe & plot distribution
    stats_df = compute_dataset_stats(dataset_dir)
    plot_class_distribution(
        stats_df,
        save_path=config.OUTPUT_DIR / "class_distribution.png"
    )

    # 4. Dimension inspection
    inspect_image_dimensions(dataset_dir, num_samples=60)

    # 5. Plot sample X-rays
    plot_sample_images(
        dataset_dir,
        n_samples_per_class=5,
        save_path=config.OUTPUT_DIR / "sample_xrays_grid.png"
    )

    # 6. Print class imbalance explanation
    train_counts = counts.get("train", {})
    explain_class_imbalance(
        train_normal=train_counts.get("NORMAL", 1341),
        train_pneumonia=train_counts.get("PNEUMONIA", 3875)
    )

    print("\n[SUCCESS] Exploratory Data Analysis complete! Plots saved in:", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
