"""
Visualization utilities: training curves, confusion matrix, radar charts, and sample grids.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


def plot_training_curves(history: Dict[str, List[float]], save_path: str):
    """
    Plots Train vs Validation Loss & Accuracy curves over epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2, markersize=4)
    plt.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2, markersize=4)
    plt.title("Biểu đồ Loss qua các Epochs", fontsize=14, fontweight="bold", pad=10)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc", linewidth=2, markersize=4)
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "g--s", label="Val Acc", linewidth=2, markersize=4)
    plt.title("Biểu đồ Độ chính xác (Accuracy %)", fontsize=14, fontweight="bold", pad=10)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    """
    Generates and saves a styled Confusion Matrix heatmap.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("Ma trận Nhầm lẫn (Confusion Matrix)", fontsize=15, fontweight="bold", pad=15)
    plt.xlabel("Nhãn dự đoán (Predicted Label)", fontsize=12, labelpad=10)
    plt.ylabel("Nhãn thực tế (True Label)", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_radar_chart(categories: List[str], values: List[float], player_name: str, save_path: Optional[str] = None):
    """
    Generates FIFA-style hexagonal Radar Chart for player attributes.
    """
    # Clean category names for display (take the short English title)
    clean_cats = [c.split(" / ")[0] for c in categories]
    
    num_vars = len(clean_cats)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Close the loop
    plot_values = values + [values[0]]
    plot_angles = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw background polygon rings
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles, clean_cats, color="#1e293b", size=11, weight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="#64748b", size=9)
    plt.ylim(0, 100)

    # Plot data
    ax.plot(plot_angles, plot_values, color="#2563eb", linewidth=2.5, linestyle="solid")
    ax.fill(plot_angles, plot_values, color="#3b82f6", alpha=0.35)

    plt.title(f"Chỉ số năng lực: {player_name}", size=14, weight="bold", pad=20, color="#0f172a")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    return fig
