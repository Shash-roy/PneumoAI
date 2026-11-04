"""
Dataset Management, Preprocessing, and Stratified Split Pipeline.
==================================================================
Handles:
1. Intelligent dataset path auto-detection (Kaggle Notebooks, Google Colab, Local).
2. Fixing the Kaggle validation split flaw (only 16 images!) by creating a stratified
   85:15 split from combined train + val sets while leaving test untouched.
3. Preprocessing: 3-channel RGB conversion, 224x224 resizing, ImageNet normalization.
4. Light medical-safe augmentation on training images only.
5. PyTorch Datasets and DataLoaders generation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.config import config


def find_dataset_dir(candidate_roots: Optional[List[str]] = None) -> Path:
    """
    Intelligently searches common environment directories to locate the chest_xray folder.
    Supports:
      - Kaggle Notebooks: /kaggle/input/chest-xray-pneumonia/...
      - Google Colab: /content/chest_xray or /content/chest-xray-pneumonia
      - Local Project: ./data/chest_xray or ./chest_xray
      
    Returns:
        Path to the root directory containing 'train', 'val', and 'test'.
        
    Raises:
        FileNotFoundError: If no valid dataset directory is found.
    """
    default_candidates = [
        # Kaggle input paths
        Path("/kaggle/input/chest-xray-pneumonia/chest_xray"),
        Path("/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray"),
        Path("/kaggle/input/chest-xray-pneumonia"),
        # Google Colab paths
        Path("/content/chest_xray/chest_xray"),
        Path("/content/chest_xray"),
        Path("/content/chest-xray-pneumonia"),
        # Local workspace paths
        config.DATA_DIR / "chest_xray",
        config.DATA_DIR / "chest_xray" / "chest_xray",
        config.PROJECT_ROOT / "chest_xray",
        config.PROJECT_ROOT / "chest_xray" / "chest_xray",
        Path.cwd() / "chest_xray",
        Path.cwd() / "data" / "chest_xray",
    ]
    
    if candidate_roots:
        custom_candidates = [Path(p) for p in candidate_roots]
        default_candidates = custom_candidates + default_candidates

    for candidate in default_candidates:
        if candidate.exists() and candidate.is_dir():
            # Check for required train, val, test subdirectories
            has_train = (candidate / "train").exists()
            has_val = (candidate / "val").exists()
            has_test = (candidate / "test").exists()
            
            if has_train and has_val and has_test:
                print(f"[FOUND] Valid dataset directory found at: {candidate.resolve()}")
                return candidate.resolve()
                
            # Sometimes Kaggle extracts into a nested chest_xray subfolder
            nested = candidate / "chest_xray"
            if nested.exists() and (nested / "train").exists() and (nested / "test").exists():
                print(f"[FOUND] Valid nested dataset directory found at: {nested.resolve()}")
                return nested.resolve()

    # If we reached here, the dataset was not detected. Provide clear actionable instructions.
    error_msg = (
        "\n================================================================================\n"
        "ERROR: Chest X-Ray dataset directory was NOT found!\n"
        "================================================================================\n"
        "How to provide the dataset:\n\n"
        "1. KAGGLE NOTEBOOK:\n"
        "   - Click '+ Add Input' in the right sidebar.\n"
        "   - Search for: 'paultimothymooney/chest-xray-pneumonia'\n"
        "   - Click 'Add' and re-run.\n\n"
        "2. GOOGLE COLAB:\n"
        "   - Run the automated download cell or run scripts/download_dataset.py\n"
        "   - Authenticate with your Kaggle API key (stored in Colab Secrets or kaggle.json)\n\n"
        "3. LOCAL WORKSPACE:\n"
        "   - Download from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia\n"
        "   - Extract into: PneumoniaML/data/chest_xray/\n"
        "   - Ensure directory contains: 'train/', 'val/', 'test/'\n"
        "================================================================================\n"
    )
    raise FileNotFoundError(error_msg)


def verify_dataset_structure(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """
    Verifies that all required folders exist and counts the images.
    Returns a dictionary of counts: {'train': {...}, 'val': {...}, 'test': {...}}
    """
    subdirs = ["train", "val", "test"]
    classes = config.CLASSES
    counts = {}

    for s in subdirs:
        s_dir = data_dir / s
        if not s_dir.exists():
            raise FileNotFoundError(f"Missing required subdirectory: {s_dir}")
        counts[s] = {}
        for c in classes:
            c_dir = s_dir / c
            if not c_dir.exists():
                raise FileNotFoundError(f"Missing required class folder: {c_dir}")
            # Count valid image files
            images = [
                f for f in c_dir.iterdir()
                if f.suffix.lower() in [".jpeg", ".jpg", ".png"]
            ]
            counts[s][c] = len(images)
            
    print("[VERIFIED] Dataset structure is valid.")
    for s, c_counts in counts.items():
        total = sum(c_counts.values())
        print(f"  - {s.upper():<5} : Total={total:<4} | NORMAL={c_counts.get('NORMAL', 0):<4} | PNEUMONIA={c_counts.get('PNEUMONIA', 0):<4}")
        
    return counts


class RGBConvert(object):
    """
    Converts single-channel grayscale chest X-rays to 3-channel RGB format.
    Pretrained ResNet-18 expects 3 color channels (ImageNet weights).
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs PyTorch image transformations.
    
    Medical Augmentation Philosophy:
    Medical images require extreme care with data augmentation!
    - Rotation: Small angle (+/- 10 degrees) mimics slight patient tilt.
    - Zoom/Crop: Scale 0.90 - 1.00 mimics minor distance variations from the X-ray detector.
    - Brightness/Contrast: +/- 10% mimics slight radiographic exposure variance.
    - NEVER apply vertical flip (the heart and diaphragm orientation is strictly anatomical).
    - Validation and test images receive NO random augmentation; only deterministic resize & norm.
    """
    train_transform = transforms.Compose([
        RGBConvert(),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomResizedCrop(
            config.IMAGE_SIZE,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05)
        ),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    val_test_transform = transforms.Compose([
        RGBConvert(),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    return train_transform, val_test_transform


class TransformSubset(Dataset):
    """
    A Dataset wrapper that applies a specific transform pipeline to a PyTorch Subset.
    This enables taking an underlying ImageFolder and applying training transforms
    to the train subset and validation transforms to the val subset without reloading files.
    """
    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        if self.transform is not None:
            # If subset returns an un-transformed PIL image
            if isinstance(x, Image.Image):
                x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)


class CustomImageFolder(Dataset):
    """
    Custom Image Dataset holding image paths and labels.
    Applies transforms on the fly upon indexing.
    """
    def __init__(self, image_paths: List[Path], targets: List[int], transform: transforms.Compose):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        target = self.targets[idx]
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self) -> int:
        return len(self.image_paths)


def create_stratified_train_val_split(
    data_dir: Path,
    seed: int = config.SEED
) -> Tuple[CustomImageFolder, CustomImageFolder, datasets.ImageFolder, Dict[str, int]]:
    """
    Solves the critical Kaggle dataset design flaw:
    - The original Kaggle 'val' folder contains only 16 images (8 Normal, 8 Pneumonia).
    - 16 images is far too small to compute a statistically reliable validation loss or ROC-AUC.
    - Solution:
        1. Combine all images from original 'train' and 'val' folders (~5,232 images).
        2. Create a stratified 85:15 split (85% train, 15% val).
        3. Keep the official 'test' folder completely untouched!
        
    Why keep the test set untouched?
    Using test set images to tune hyperparameters, pick model checkpoints, or optimize
    decision thresholds causes 'data snooping' / data leakage. It produces overly
    optimistic performance estimates that fail when deployed on real patients.
    """
    train_transform, val_transform = get_transforms()
    
    # 1. Gather all file paths and labels from original train and val
    all_paths: List[Path] = []
    all_labels: List[int] = []
    
    for split_name in ["train", "val"]:
        split_dir = data_dir / split_name
        for cls_name, cls_idx in config.CLASS_TO_IDX.items():
            cls_dir = split_dir / cls_name
            for img_file in cls_dir.iterdir():
                if img_file.suffix.lower() in [".jpeg", ".jpg", ".png"]:
                    all_paths.append(img_file)
                    all_labels.append(cls_idx)
                    
    all_paths_arr = np.array(all_paths)
    all_labels_arr = np.array(all_labels)
    
    # 2. Perform Stratified 85:15 Split
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.VAL_RATIO,
        random_state=seed
    )
    train_idx, val_idx = next(sss.split(all_paths_arr, all_labels_arr))
    
    train_paths = all_paths_arr[train_idx].tolist()
    train_labels = all_labels_arr[train_idx].tolist()
    
    val_paths = all_paths_arr[val_idx].tolist()
    val_labels = all_labels_arr[val_idx].tolist()
    
    # Create dataset instances
    train_dataset = CustomImageFolder(train_paths, train_labels, transform=train_transform)
    val_dataset = CustomImageFolder(val_paths, val_labels, transform=val_transform)
    
    # 3. Load official test folder with standard ImageFolder (deterministic evaluation)
    test_dir = data_dir / "test"
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=val_transform)
    
    # Calculate counts
    n_train_normal = train_labels.count(config.CLASS_TO_IDX["NORMAL"])
    n_train_pneumonia = train_labels.count(config.CLASS_TO_IDX["PNEUMONIA"])
    n_val_normal = val_labels.count(config.CLASS_TO_IDX["NORMAL"])
    n_val_pneumonia = val_labels.count(config.CLASS_TO_IDX["PNEUMONIA"])
    
    split_info = {
        "train_total": len(train_dataset),
        "train_normal": n_train_normal,
        "train_pneumonia": n_train_pneumonia,
        "val_total": len(val_dataset),
        "val_normal": n_val_normal,
        "val_pneumonia": n_val_pneumonia,
        "test_total": len(test_dataset)
    }
    
    print("\n[SPLIT SUMMARY] Stratified 85:15 Split Created:")
    print(f"  - Train Set : Total = {len(train_dataset)} (NORMAL: {n_train_normal}, PNEUMONIA: {n_train_pneumonia})")
    print(f"  - Val Set   : Total = {len(val_dataset)}   (NORMAL: {n_val_normal}, PNEUMONIA: {n_val_pneumonia})")
    print(f"  - Test Set  : Total = {len(test_dataset)}  (Untouched official Kaggle test set)")
    
    return train_dataset, val_dataset, test_dataset, split_info


def get_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test datasets.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

