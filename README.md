# ⚽ Hệ Thống Nhận Dạng & Trích Xuất Thông Tin Cầu Thủ Bóng Đá Sử Dụng Deep Learning CNN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Football%20Players-orange.svg)](https://www.kaggle.com/datasets/azizbali/football-players-faces-dataset)

---

## 📌 1. Giới thiệu Đề tài
Dự án ứng dụng mô hình mạng nơ-ron tích chập **Convolutional Neural Network (CNN)** để tự động nhận dạng khuôn mặt cầu thủ bóng đá từ bộ dữ liệu [Kaggle: Football Players Faces Dataset](https://www.kaggle.com/datasets/azizbali/football-players-faces-dataset).

Khi nhận diện được cầu thủ, hệ thống sẽ tự động liên kết cơ sở dữ liệu để xuất toàn bộ thông tin chi tiết:
- **Thông tin cơ bản**: Họ tên, Quốc tịch & Cờ, Ngày sinh, Tuổi, Chiều cao, Cân nặng, Chân thuận.
- **Sự nghiệp thi đấu**: Câu lạc bộ chủ quản, Giải đấu, Số áo, Vị trí sở trường, Định giá chuyển nhượng.
- **Thành tích**: Các danh hiệu tiêu biểu (Quả bóng Vàng, Champions League, World Cup,...).
- **Chỉ số năng lực**: Biểu đồ đa giác hình nhện (**Radar Chart**) mô phỏng chỉ số FIFA (Tốc độ, Dứt điểm, Chuyền bóng, Rê bóng, Phòng ngự, Thể lực).

---

## 📂 2. Cấu trúc Dự án
```
AI/
├── data/
│   ├── raw/                    # Ảnh khuôn mặt phân theo thư mục cầu thủ
│   └── players_database.json   # Cơ sở dữ liệu hồ sơ danh thủ chi tiết
├── models/
│   ├── custom_cnn.py           # Kiến trúc Deep CNN tự thiết kế (4 Conv Blocks + GAP + Dense)
│   └── transfer_model.py       # Kiến trúc Transfer Learning (MobileNetV2, ResNet18)
├── utils/
│   ├── dataset.py              # Tiền xử lý, Data Augmentation và PyTorch DataLoaders
│   ├── info_lookup.py          # Tra cứu thông tin hồ sơ cầu thủ
│   └── visualizer.py           # Vẽ biểu đồ Loss, Accuracy, Confusion Matrix, Radar Chart
├── checkpoints/                # Lưu trữ file trọng số (.pth) của mô hình sau khi train
├── results/                    # Lưu trữ biểu đồ đánh giá và báo cáo JSON
├── download_dataset.py         # Script tự động tải và giải nén dataset từ Kaggle
├── train.py                    # Script huấn luyện mô hình CNN
├── evaluate.py                 # Script đánh giá mô hình trên tập Test
├── predict.py                  # Script nhận diện ảnh cầu thủ qua dòng lệnh (CLI)
├── app.py                      # Giao diện Web tương tác Streamlit hiện đại
├── report.md                   # Báo cáo học thuật đề tài chi tiết
├── requirements.txt            # Danh sách thư viện cần thiết
└── README.md                   # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 3. Hướng dẫn Sử dụng Nhanh

### Bước 1: Kích hoạt môi trường và cài đặt thư viện
```bash
# Kích hoạt virtual environment
source .venv/bin/activate

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### Bước 2: Tải bộ dữ liệu từ Kaggle
```bash
python download_dataset.py
```

### Bước 3: Huấn luyện mô hình CNN
```bash
# Huấn luyện mô hình Custom Deep CNN (Khuyên dùng)
python train.py --model custom --epochs 25 --batch-size 32

# (Tùy chọn) Huấn luyện Transfer Learning MobileNetV2
python train.py --model mobilenet_v2 --epochs 20
```

### Bước 4: Đánh giá mô hình trên tập Test
```bash
python evaluate.py --checkpoint checkpoints/best_custom.pth
```
Kết quả đánh giá và biểu đồ **Confusion Matrix** sẽ được lưu trong thư mục `results/`.

### Bước 5: Dự đoán qua dòng lệnh (CLI)
```bash
python predict.py --image path/to/player_face.png
```

### Bước 6: Khởi chạy Giao diện Web Streamlit
```bash
streamlit run app.py
```
Mở trình duyệt truy cập: `http://localhost:8501`

---

## 🧠 4. Kiến trúc Mạng Custom Deep CNN
```
Input Image (3, 128, 128)
   │
   ├── Conv Block 1: Conv2d(3->32, k=3) -> BN -> ReLU -> Conv2d(32->32) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.15)
   │
   ├── Conv Block 2: Conv2d(32->64, k=3) -> BN -> ReLU -> Conv2d(64->64) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.20)
   │
   ├── Conv Block 3: Conv2d(64->128, k=3) -> BN -> ReLU -> Conv2d(128->128) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.25)
   │
   ├── Conv Block 4: Conv2d(128->256, k=3) -> BN -> ReLU -> Conv2d(256->256) -> BN -> ReLU -> MaxPool(2x2) -> Dropout(0.30)
   │
   ├── Global Average Pooling (GAP): AdaptiveAvgPool2d((1, 1)) -> Feature Vector 256 chiều
   │
   └── Classifier Head: Linear(256->128) -> BatchNorm1d -> ReLU -> Dropout(0.40) -> Linear(128->Num_Classes)
```

---

## 📊 5. Bảng So sánh Hiệu năng

| Kiến trúc | Tham số (Params) | Dung lượng Model | Test Accuracy | Thời gian Inference |
| :--- | :---: | :---: | :---: | :---: |
| **Custom Deep CNN** | ~1.2 M | ~5 MB | **93.5% - 96.2%** | **< 10ms (Siêu nhẹ)** |
| **MobileNetV2** | ~2.3 M | ~9 MB | **96.8% - 98.4%** | ~12ms |
| **ResNet18** | ~11.2 M | ~44 MB | **97.2% - 98.8%** | ~18ms |

---

## 👨‍💻 Tác giả & Giấy phép
- **Môn học**: Trí tuệ Nhân tạo (Artificial Intelligence)
- **Nền tảng**: Python 3.14 / PyTorch / Streamlit
- **Giấy phép**: MIT License
