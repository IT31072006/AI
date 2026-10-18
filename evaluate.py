"""
Model evaluation script for Football Player Face Recognition.
Computes Accuracy, Precision, Recall, F1-Score, and generates Confusion Matrix.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from models.custom_cnn import build_custom_cnn
from models.transfer_model import build_transfer_model
from utils.dataset import create_data_loaders
from utils.visualizer import plot_confusion_matrix


def evaluate_model(checkpoint_path: str, data_dir: str = "data/raw", results_dir: str = "results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(" 📊 EVALUATING FOOTBALL PLAYER FACE RECOGNITION MODEL")
    print("=" * 65)
    print(f"📦 Checkpoint: {checkpoint_path}")
    print(f"💻 Device: {device}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file '{checkpoint_path}' does not exist!")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names")
    model_type = checkpoint.get("model_type", "custom")
    img_size = checkpoint.get("img_size", 128)
    num_classes = len(class_names)

    print(f"🧠 Model Architecture: {model_type.upper()}")
    print(f"👥 Classes ({num_classes}): {', '.join(class_names)}")

    # Load test dataloader
    _, _, test_loader, _ = create_data_loaders(
        root_dir=data_dir,
        img_size=img_size,
        batch_size=32
    )
    print(f"🧪 Test set size: {len(test_loader.dataset)} images")

    # Instantiate model
    if model_type == "custom":
        model = build_custom_cnn(num_classes=num_classes)
    else:
        model = build_transfer_model(model_name=model_type, num_classes=num_classes, pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print("\n" + "=" * 65)
    print(" 🎯 CLASSIFICATION REPORT (ĐÁNH GIÁ CHI TIẾT THEO CẦU THỦ)")
    print("=" * 65)
    print(report_str)
    print(f"⭐ OVERALL TEST ACCURACY: {acc * 100:.2f}%")
    print("=" * 65)

    # Plot Confusion Matrix
    os.makedirs(results_dir, exist_ok=True)
    cm_path = os.path.join(results_dir, f"confusion_matrix_{model_type}.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    print(f"🖼️ Confusion Matrix plot saved to: {cm_path}")

    # Save detailed JSON evaluation
    eval_json_path = os.path.join(results_dir, f"evaluation_{model_type}.json")
    eval_data = {
        "model_type": model_type,
        "checkpoint": checkpoint_path,
        "test_accuracy": acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"📝 Full evaluation metrics saved to: {eval_json_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CNN model on test dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_custom.pth", help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.data_dir, args.results_dir)
