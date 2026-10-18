"""
Script to download and organize the Football Players Faces Dataset from Kaggle.
Dataset: azizbali/football-players-faces-dataset (2,247 images across 8 players)
"""

import os
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_NAME = "azizbali/football-players-faces-dataset"


def print_summary(target_dir: Path):
    """Print an ASCII summary of classes and image counts."""
    print("\n" + "=" * 60)
    print(" 📊 DATASET INVENTORY SUMMARY (FOOTBALL PLAYERS FACES)")
    print("=" * 60)
    
    classes = sorted([d for d in target_dir.iterdir() if d.is_dir()])
    total_images = 0
    
    if not classes:
        print("❌ No class directories found in", target_dir)
        return
        
    print(f"{'Class / Player Name':<30} | {'Images Count':<15}")
    print("-" * 60)
    
    for c in classes:
        images = list(c.glob("*.png")) + list(c.glob("*.jpg")) + list(c.glob("*.jpeg"))
        count = len(images)
        total_images += count
        print(f"⚽ {c.name:<28} | {count:<15}")
        
    print("-" * 60)
    print(f"🏆 TOTAL CLASSES: {len(classes)} | TOTAL IMAGES: {total_images}")
    print("=" * 60 + "\n")


def organize_from_cache(cache_dir: Path):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Fifa_player_dataset (Courtois, Dybala, Kross, Pogba)
    fifa_dir = cache_dir / "Football players faces dataset" / "Fifa_player_dataset"
    if fifa_dir.exists():
        for d in fifa_dir.iterdir():
            if d.is_dir():
                dest = RAW_DIR / d.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(d, dest)
                
    # 2. Messi parent folder (Cristiano Ronaldo, Lionel Messi, Mohamed Salah, Neymar)
    messi_parent = cache_dir / "Lionel Messi"
    if messi_parent.exists():
        for d in messi_parent.iterdir():
            if d.is_dir():
                dest = RAW_DIR / d.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(d, dest)


def setup_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if len(existing_dirs) >= 8:
        print(f"✅ Dataset already organized in {RAW_DIR}")
        print_summary(RAW_DIR)
        return

    # Check kagglehub cache directory
    cache_path = Path("/home/quan/.cache/kagglehub/datasets/azizbali/football-players-faces-dataset/versions/2")
    if cache_path.exists():
        print("📦 Organizing from local cache...")
        organize_from_cache(cache_path)
        print_summary(RAW_DIR)
        return

    print(f"⬇️ Downloading dataset '{DATASET_NAME}' via kagglehub...")
    import kagglehub
    download_path = kagglehub.dataset_download(DATASET_NAME)
    organize_from_cache(Path(download_path))
    print_summary(RAW_DIR)


if __name__ == "__main__":
    setup_dataset()
