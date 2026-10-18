"""
Main training pipeline for Football Player Face Recognition.
Trains either Custom CNN or Transfer Learning (MobileNetV2 / ResNet18).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np

from models.custom_cnn import build_custom_cnn
from models.transfer_model import build_transfer_model
from utils.dataset import create_data_loaders
from utils.visualizer import plot_training_curves


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Football Player Face Recognition CNN")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "mobilenet_v2", "resnet18"],
                        help="Model architecture ('custom', 'mobilenet_v2', 'resnet18')")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 0.001 for custom, 0.0003 for transfer)")
    parser.add_argument("--img-size", type=int, default=128, help="Input image resolution")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save metrics and graphs")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--freeze-backbone", action="store_true", default=False,
                        help="Freeze backbone for transfer learning (much faster on CPU)")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(f"🚀 FOOTBALL PLAYER FACE RECOGNITION - TRAINING PIPELINE")
    print("=" * 65)
    print(f"💻 Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"🧠 Model Architecture: {args.model.upper()}")
    print(f"📂 Dataset Directory: {args.data_dir}")

    # Learning rate defaults
    if args.lr is None:
        args.lr = 0.001 if args.model == "custom" else 0.0005
    print(f"⚙️  Hyperparameters: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}, ImgSize={args.img_size}x{args.img_size}")

    # Load Data
    try:
        train_loader, val_loader, test_loader, class_names = create_data_loaders(
            root_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Hint: Ensure you have run 'python download_dataset.py' to download the dataset first.")
        sys.exit(1)

    num_classes = len(class_names)
    print(f"👥 Classes ({num_classes}): {', '.join(class_names)}")
    print(f"📊 Dataset Splits: Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)} | Test={len(test_loader.dataset)}")
    print("-" * 65)

    # Calculate balanced class weights to solve minority class underrepresentation (e.g., Neymar ~69 vs Messi ~337)
    train_labels = train_loader.dataset.labels
    class_counts = np.bincount(train_labels, minlength=num_classes)
    total_samples = len(train_labels)
    class_weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(float))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("⚖️  Balanced Class Distribution & Loss Weights:")
    for name, count, weight in zip(class_names, class_counts, class_weights):
        print(f"   • {name:<18}: {count:>4} train samples (Weight: {weight:.2f}x)")
    print("-" * 65)

    # Save class names mapping
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    classes_path = os.path.join(args.checkpoints_dir, "classes.json")
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"💾 Saved class index mapping to: {classes_path}")

    # Initialize Model
    if args.model == "custom":
        model = build_custom_cnn(num_classes=num_classes)
    else:
        model = build_transfer_model(
            model_name=args.model,
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=args.freeze_backbone
        )
        
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 Trainable Parameters: {num_params:,}")

    # Loss with balanced weights, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Tracking metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(args.checkpoints_dir, f"best_{args.model}.pth")

    start_time = time.time()

    print("\n" + f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^12} | {'Val Acc':^11} | {'LR':^10}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Check for improvement
        is_best = val_acc > best_val_acc
        star = " ⭐ [BEST]" if is_best else ""
        
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "model_type": args.model,
                "img_size": args.img_size
            }, best_model_path)
        else:
            patience_counter += 1

        epoch_duration = time.time() - epoch_start
        print(f"{epoch:^7d} | {train_loss:^12.4f} | {train_acc*100:^10.2f}% | {val_loss:^12.4f} | {val_acc*100:^10.2f}% | {current_lr:^10.6f}{star} ({epoch_duration:.1f}s)", flush=True)

        # Early Stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs without improvement!")
            break

    total_training_time = time.time() - start_time
    print("-" * 75)
    print(f"🏁 Training finished in {total_training_time / 60:.2f} minutes.")
    print(f"🏆 Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"💾 Best checkpoint saved to: {best_model_path}")

    # Plot & save curves
    plot_path = os.path.join(args.results_dir, f"training_curves_{args.model}.png")
    plot_training_curves(history, plot_path)
    print(f"📈 Training curves saved to: {plot_path}")

    # Save history json
    history_path = os.path.join(args.results_dir, f"history_{args.model}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"📝 Training history saved to: {history_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
