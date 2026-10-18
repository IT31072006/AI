"""
CLI Prediction Script: Predict player from face image and display detailed player profile.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from models.custom_cnn import build_custom_cnn
from models.transfer_model import build_transfer_model
from utils.info_lookup import get_player_info, get_radar_stats
from utils.visualizer import plot_radar_chart
from utils.dataset import NORM_MEAN, NORM_STD
from utils.face_detector import detect_and_crop_face


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names")
    model_type = checkpoint.get("model_type", "custom")
    img_size = checkpoint.get("img_size", 128)

    if model_type == "custom":
        model = build_custom_cnn(num_classes=len(class_names))
    else:
        model = build_transfer_model(model_name=model_type, num_classes=len(class_names), pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names, img_size


def predict_player(image_path: str, checkpoint_path: str = "checkpoints/best_custom.pth", output_radar: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if checkpoint_path == "ensemble":
        model_c, class_names, img_size = load_model_from_checkpoint("checkpoints/best_custom.pth", device)
        model_m, _, _ = load_model_from_checkpoint("checkpoints/best_mobilenet_v2.pth", device)
        is_ensemble = True
    else:
        model, class_names, img_size = load_model_from_checkpoint(checkpoint_path, device)
        is_ensemble = False

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    image = Image.open(image_path).convert("RGB")
    cropped_image, detected, bbox = detect_and_crop_face(image)
    if detected:
        print(f"🎯 Face Auto-Crop: Phát hiện khuôn mặt tại {bbox}, tự động cắt chuẩn 128x128.")
    tensor = transform(cropped_image).unsqueeze(0).to(device)

    with torch.no_grad():
        if is_ensemble:
            out_c = F.softmax(model_c(tensor), dim=1)
            out_m = F.softmax(model_m(tensor), dim=1)
            probs = (0.35 * out_c + 0.65 * out_m)[0].cpu().numpy()
        else:
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    # Sort probabilities
    top_indices = probs.argsort()[::-1]
    top_class_idx = top_indices[0]
    predicted_class = class_names[top_class_idx]
    confidence = probs[top_class_idx] * 100

    # Retrieve Player Information
    info = get_player_info(predicted_class)

    # Print Formatted Profile Card
    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║ {'⚽ HỆ THỐNG NHẬN DẠNG & THÔNG TIN CẦU THỦ BÓNG ĐÁ':^66} ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║ 🎯 DỰ ĐOÁN: {info['name']:<30} (Độ tin cậy: {confidence:>5.2f}%) ║")
    print("╟" + "─" * 68 + "╢")
    print(f"║ 👤 Họ và tên:      {info.get('full_name', info['name']):<48} ║")
    print(f"║ 🌍 Quốc tịch:      {info.get('flag', '')} {info.get('nationality', 'N/A'):<45} ║")
    print(f"║ 🛡️  Câu lạc bộ:     {info.get('club', 'N/A'):<48} ║")
    print(f"║ 🏆 Giải đấu:       {info.get('league', 'N/A'):<48} ║")
    print(f"║ 📍 Vị trí:         {info.get('position', 'N/A'):<48} ║")
    print(f"║ 🔢 Số áo:          #{info.get('jersey_number', 0):<47} ║")
    print(f"║ 🎂 Tuổi:           {info.get('age', 'N/A')} tuổi (Sinh: {info.get('date_of_birth', 'N/A')}){'':<26} ║")
    print(f"║ 📏 Chiều cao/Nặng: {info.get('height', 'N/A')} / {info.get('weight', 'N/A'):<35} ║")
    print(f"║ 👟 Thuận chân:     {info.get('preferred_foot', 'N/A'):<48} ║")
    print(f"║ 💰 Giá trị CN:     {info.get('market_value', 'N/A'):<48} ║")
    print(f"║ ⭐ FIFA Rating:    {info.get('fifa_rating', 'N/A')}/100{'':<43} ║")
    print("╟" + "─" * 68 + "╢")
    print("║ 🏅 DANH HIỆU TIÊU BIỂU:                                            ║")
    for trophy in info.get("trophies", [])[:4]:
        print(f"║   • {trophy:<62} ║")
    print("╟" + "─" * 68 + "╢")
    print("║ 📊 TOP 3 DỰ ĐOÁN XÁC SUẤT CAO NHẤT:                                ║")
    for i in range(min(3, len(top_indices))):
        idx = top_indices[i]
        c_name = class_names[idx]
        p_val = probs[idx] * 100
        bar = "█" * int(p_val // 4)
        print(f"║   {i+1}. {c_name:<15} {p_val:>6.2f}% {bar:<40} ║")
    print("╚" + "═" * 68 + "╝\n")

    if output_radar:
        radar_path = "results/predicted_player_radar.png"
        categories, values = get_radar_stats(predicted_class)
        plot_radar_chart(categories, values, info['name'], save_path=radar_path)
        print(f"📈 Radar chart saved to: {radar_path}")

    return predicted_class, confidence, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict football player from face image")
    parser.add_argument("--image", type=str, required=True, help="Path to face image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_custom.pth", help="Path to trained model checkpoint")
    args = parser.parse_args()

    predict_player(args.image, args.checkpoint)
