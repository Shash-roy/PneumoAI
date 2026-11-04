"""
Secure Dataset Downloader and Setup Assistant.
=============================================
Safely acquires and unpacks the Kaggle 'paultimothymooney/chest-xray-pneumonia' dataset.

Features:
- Never hardcodes or exposes API tokens or credentials in source code.
- Checks if dataset is already downloaded.
- In Google Colab: guides secure authentication via google.colab.userdata or uploaded kaggle.json.
- In Kaggle Notebooks: auto-detects Kaggle's /kaggle/input directory.
- On Local machines: verifies ~/.kaggle/kaggle.json or provides clean download instructions.
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config


def download_via_kaggle_api(dest_dir: Path) -> bool:
    """
    Attempts to download the dataset using the Kaggle Python API.
    Ensures credentials are properly configured without exposing them.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if in Google Colab and credentials are in Colab Secrets
    try:
        from google.colab import userdata
        kaggle_username = userdata.get('KAGGLE_USERNAME')
        kaggle_key = userdata.get('KAGGLE_KEY')
        if kaggle_username and kaggle_key:
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
            print("[COLAB] Authenticated using Colab Secrets.")
    except Exception:
        pass

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"[KAGGLE API] Authenticated successfully. Downloading dataset: {config.KAGGLE_DATASET_ID}...")
        
        api.dataset_download_files(
            dataset=config.KAGGLE_DATASET_ID,
            path=str(dest_dir),
            unzip=True,
            quiet=False
        )
        print(f"[SUCCESS] Dataset downloaded and extracted to: {dest_dir}")
        return True
    except Exception as e:
        print(f"[NOTICE] Kaggle API automated download could not be completed: {e}")
        return False


def main():
    print("=" * 80)
    print("CHEST X-RAY DATASET ACQUISITION & VERIFICATION ASSISTANT")
    print("=" * 80)
    
    dest_dir = config.DATA_DIR / "chest_xray"
    
    # 1. Check if already exists
    if dest_dir.exists() and (dest_dir / "train").exists() and (dest_dir / "test").exists():
        print(f"[ALREADY EXISTS] Dataset is already present at: {dest_dir.resolve()}")
        return

    # 2. Try Kaggle API download
    print(f"[CHECK] Target directory: {dest_dir}")
    print("[INFO] Attempting download using Kaggle API...")
    success = download_via_kaggle_api(dest_dir)
    
    if not success:
        print("\n" + "=" * 80)
        print("MANUAL SETUP INSTRUCTIONS")
        print("=" * 80)
        print("Option A (Recommended for Google Colab):")
        print("  1. In Colab, click the Key icon (Secrets) on the left.")
        print("  2. Add 'KAGGLE_USERNAME' and 'KAGGLE_KEY' from your Kaggle Account settings.")
        print("  3. Grant access to notebook and run this script again.\n")
        print("Option B (Kaggle Notebooks):")
        print("  1. Click '+ Add Input' on the right panel.")
        print("  2. Search for: 'paultimothymooney/chest-xray-pneumonia'")
        print("  3. Add input — the codebase auto-detects /kaggle/input automatically!\n")
        print("Option C (Local Machine):")
        print("  1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
        print("  2. Download 'archive.zip' and extract its contents into:")
        print(f"     {dest_dir.resolve()}")
        print("=" * 80)


if __name__ == "__main__":
    main()
