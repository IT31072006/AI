"""
Dataset utilities, augmentations, and PyTorch DataLoaders for Player Face Recognition.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np


# Standard ImageNet normalization statistics
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 128) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns data transforms for training (with data augmentation) and validation/testing.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    return train_transform, eval_transform


class FootballPlayerDataset(Dataset):
    """
    Custom PyTorch Dataset for Football Player Faces.
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback black image if corrupt
            image = Image.new("RGB", (128, 128), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def scan_dataset(root_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Scans directory for subdirectories (classes) and image files.
    Returns image paths, numeric labels, and class names list.
    """
    root_path = Path(root_dir)
    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for class_dir in class_dirs:
        for file in class_dir.iterdir():
            if file.suffix.lower() in valid_exts:
                image_paths.append(str(file))
                labels.append(class_to_idx[class_dir.name])
                
    return image_paths, labels, class_names


def create_data_loaders(
    root_dir: str,
    img_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 2,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
    balanced_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates train, val, and test DataLoaders with stratified splitting and balanced sampling.
    """
    image_paths, labels, class_names = scan_dataset(root_dir)
    
    if len(image_paths) == 0:
        raise ValueError(f"No valid images found in dataset directory '{root_dir}'!")

    test_ratio = 1.0 - train_ratio - val_ratio
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: val vs test
    val_relative_ratio = val_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1.0 - val_relative_ratio),
        random_state=random_state,
        stratify=temp_labels
    )

    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_dataset = FootballPlayerDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = FootballPlayerDataset(val_paths, val_labels, transform=eval_transform)
    test_dataset = FootballPlayerDataset(test_paths, test_labels, transform=eval_transform)

    # Use WeightedRandomSampler to solve class imbalance (e.g., Neymar ~48 vs Messi ~235)
    if balanced_sampling:
        class_sample_counts = np.bincount(train_labels)
        class_weights = 1.0 / np.maximum(class_sample_counts, 1).astype(float)
        samples_weight = np.array([class_weights[label] for label in train_labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_names
