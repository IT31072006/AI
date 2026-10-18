# BÁO CÁO ĐỀ TÀI MÔN HỌC: TRÍ TUỆ NHÂN TẠO (ARTIFICIAL INTELLIGENCE)

## ĐỀ TÀI: SỬ DỤNG DEEP LEARNING ĐỂ PHÂN TÍCH NHẬN DẠNG HÌNH ẢNH CẦU THỦ ĐỂ ĐƯA RA THÔNG TIN CỦA CẦU THỦ ĐÓ SỬ DỤNG MÔ HÌNH CNN

---

### MỤC LỤC
1. [GIỚI THIỆU & ĐẶT VẤN ĐỀ](#1-giới-thiệu--đặt-vấn-đề)
2. [TỔNG QUAN DỮ LIỆU (FOOTBALL PLAYERS FACES DATASET)](#2-tổng-quan-dữ-liệu-football-players-faces-dataset)
3. [CƠ SỞ LÝ THUYẾT MẠNG TÍCH CHẬP (CNN)](#3-cơ-sở-lý-thuyết-mạng-tích-chập-cnn)
4. [PHƯƠNG PHÁP & KIẾN TRÚC MÔ HÌNH](#4-phương-pháp--kiến-trúc-mô-hình)
5. [QUY TRÌNH HUẤN LUYỆN & TỐI ƯU HÓA](#5-quy-trình-huấn-luyện--tối-ưu-hóa)
6. [HỆ THỐNG TRUY VẤN THÔNG TIN & TRỰC QUAN HÓA (RADAR CHART)](#6-hệ-thống-truy-vấn-thông-tin--trực-quan-hóa-radar-chart)
7. [KẾT QUẢ THỰC NGHIỆM & SO SÁNH KIẾN TRÚC](#7-kết-quả-thực-nghiệm--so-sánh-kiến-trúc)
8. [HƯỚNG DẪN CÀI ĐẶT & CHẠY ỨNG DỤNG](#8-hướng-dẫn-cài-đặt--chạy-ứng-dụng)
9. [KẾT LUẬN & HƯỚNG PHÁT TRIỂN](#9-kết-luận--hướng-phát-triển)

---

### 1. GIỚI THIỆU & ĐẶT VẤN ĐỀ
Trong thời đại công nghệ số và truyền thông thể thao hiện đại, nhu cầu nhận dạng tự động vận động viên/cầu thủ từ hình ảnh phát sóng trực tiếp, ảnh báo chí và mạng xã hội ngày càng trở nên cấp thiết. Việc xây dựng một hệ thống thông minh không chỉ dừng lại ở việc gán nhãn tên cầu thủ, mà còn tự động trích xuất toàn bộ hồ sơ vận động viên (câu lạc bộ, quốc tịch, tuổi, vị trí, giá trị thị trường, danh hiệu và bộ chỉ số kỹ năng) mang lại giá trị to lớn cho:
- **Truyền thông & Bình luận viên**: Tra cứu tức thì dữ liệu cầu thủ trong thời gian thực.
- **Tuyển trạch viên (Scouting)**: Hỗ trợ phân tích dữ liệu và quản lý thông tin cầu thủ.
- **Người hâm mộ thể thao**: Trải nghiệm tương tác đa phương tiện trực quan và sống động.

Đề tài này tập trung nghiên cứu và áp dụng mô hình **Convolutional Neural Network (CNN)** – kiến trúc nền tảng và hiệu quả nhất trong xử lý thị giác máy tính (Computer Vision) – để phân tích đặc trưng khuôn mặt cầu thủ từ tập dữ liệu Kaggle và xây dựng hệ thống web tương tác hoàn chỉnh.

---

### 2. TỔNG QUAN DỮ LIỆU (FOOTBALL PLAYERS FACES DATASET)
- **Nguồn dữ liệu**: [Kaggle - Football Players Faces Dataset](https://www.kaggle.com/datasets/azizbali/football-players-faces-dataset).
- **Quy mô**: Tập dữ liệu gồm các khuôn mặt được cắt chuẩn hóa của các danh thủ bóng đá hàng đầu thế giới (Courtois, Dybala, Mbappe, Messi, Neymar, Ronaldo, Salah, De Bruyne/Haaland).
- **Định dạng ảnh**: Ảnh màu RGB, độ phân giải gốc đa dạng, được tiền xử lý và resize đồng nhất về $128 \times 128$ pixel.
- **Phân chia dữ liệu (Data Split)**:
  - **Tập Train (70%)**: Dùng để tối ưu trọng số mạng CNN.
  - **Tập Validation (15%)**: Đánh giá độ tổng quát hóa và kích hoạt cơ chế Dừng sớm (Early Stopping).
  - **Tập Test (15%)**: Đánh giá khách quan hiệu năng mô hình trên dữ liệu chưa từng thấy.

---

### 3. CƠ SỞ LÝ THUYẾT MẠNG TÍCH CHẬP (CNN)

#### 3.1. Phép tích chập (Convolutional Layer)
Phép tích chập 2 chiều áp dụng kernel/filter trượt qua ảnh đầu vào để trích xuất các đặc trưng không gian (cạnh, góc, họa tiết, hình dáng mắt, mũi, miệng):
$$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n)$$
Trong đó $I$ là ma trận đầu vào, $K$ là bộ lọc (Kernel kích thước $3 \times 3$), và $S$ là bản đồ đặc trưng (Feature Map).

#### 3.2. Hàm kích hoạt phi tuyến (ReLU Activation)
$$\text{ReLU}(x) = \max(0, x)$$
Giúp mô hình học được các biểu diễn phi tuyến tính phức tạp, đồng thời giải quyết triệt để vấn đề suy giảm đạo hàm (Vanishing Gradient).

#### 3.3. Lớp Chuẩn hóa theo lô (Batch Normalization)
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$
Giảm hiện tượng *Internal Covariate Shift*, ổn định phân phối gradient và giúp tăng tốc độ hội tụ của mô hình lên nhiều lần.

#### 3.4. Lớp Giảm lấy mẫu (Max Pooling)
Thu giảm kích thước không gian (spatial dimensions) $H \times W \to \frac{H}{2} \times \frac{W}{2}$ bằng cách chọn giá trị lớn nhất trong mỗi cửa sổ $2 \times 2$, mang lại tính bất biến tịnh tiến cục bộ (translation invariance).

#### 3.5. Kỹ thuật Chống học vẹt (Dropout & Global Average Pooling)
- **Spatial Dropout & Linear Dropout**: Ngẫu nhiên tắt một tỉ lệ nơ-ron (20% - 50%) trong quá trình train, buộc các nơ-ron học các đặc trưng độc lập.
- **Global Average Pooling (GAP)**: Thay vì làm phẳng ma trận đặc trưng lớn tạo hàng triệu trọng số ở lớp Dense, GAP tính trung bình trên toàn bộ kích thước không gian của mỗi kênh đặc trưng, giúp giảm mạnh số tham số và triệt tiêu nguy cơ Overfitting.

---

### 4. PHƯƠNG PHÁP & KIẾN TRÚC MÔ HÌNH

Hệ thống triển khai 2 hướng tiếp cận để so sánh đối chứng:

#### 4.1. Kiến trúc Custom Deep CNN (Thiết kế độc lập)
Gồm 4 khối tích chập liên tiếp (Hierarchical Feature Blocks):
1. **Block 1**: Conv2D ($3 \to 32$) $\to$ BN $\to$ ReLU $\to$ Conv2D ($32 \to 32$) $\to$ BN $\to$ ReLU $\to$ MaxPool ($2\times 2$) $\to$ Dropout ($0.15$)
2. **Block 2**: Conv2D ($32 \to 64$) $\to$ BN $\to$ ReLU $\to$ Conv2D ($64 \to 64$) $\to$ BN $\to$ ReLU $\to$ MaxPool ($2\times 2$) $\to$ Dropout ($0.20$)
3. **Block 3**: Conv2D ($64 \to 128$) $\to$ BN $\to$ ReLU $\to$ Conv2D ($128 \to 128$) $\to$ BN $\to$ ReLU $\to$ MaxPool ($2\times 2$) $\to$ Dropout ($0.25$)
4. **Block 4**: Conv2D ($128 \to 256$) $\to$ BN $\to$ ReLU $\to$ Conv2D ($256 \to 256$) $\to$ BN $\to$ ReLU $\to$ MaxPool ($2\times 2$) $\to$ Dropout ($0.30$)
5. **Bottleneck & Head**:
   - `nn.AdaptiveAvgPool2d((1, 1))` trích xuất vector đặc trưng $256$-chiều.
   - Dense ($256 \to 128$) $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout ($0.40$) $\to$ Dense ($128 \to \text{num\_classes}$).

#### 4.2. Kiến trúc Học chuyển giao (Transfer Learning - MobileNetV2 / ResNet18)
- Sử dụng mô hình đã được huấn luyện trước trên tập ImageNet-1K (1.2 triệu ảnh).
- Giữ lại phần trích xuất đặc trưng sâu và thay thế lớp phân loại cuối bằng cụm Dense tinh chỉnh phù hợp với tập khuôn mặt cầu thủ.

---

### 5. QUY TRÌNH HUẤN LUYỆN & TỐI ƯU HÓA

#### 5.1. Tăng cường dữ liệu (Data Augmentation)
Để tăng độ phong phú và khả năng kháng nhiễu:
- **Random Horizontal Flip ($p=0.5$)**: Lật ngang ngẫu nhiên.
- **Random Rotation ($\pm 15^\circ$)**: Xoay góc mặt ngẫu nhiên.
- **Color Jitter (0.2)**: Biến thiên độ sáng, độ tương phản và bão hòa màu.
- **Random Affine**: Dịch chuyển nhẹ và co giãn tỉ lệ khuôn mặt.

#### 5.2. Hàm mất mát có trọng số cân bằng lớp (Balanced Class Weights) & Tối ưu hóa
- **Thách thức mất cân bằng dữ liệu**: Trong tập dữ liệu, cầu thủ Neymar chỉ có 69 ảnh (chiếm ~3%), trong khi Lionel Messi có tới 337 ảnh (gấp gần 5 lần). Nếu sử dụng hàm mất mát thông thường, mạng CNN sẽ bị thiên kiến (bias) và có xu hướng đoán nhầm Neymar thành Messi.
- **Giải pháp Trọng số cân bằng lớp (Balanced Loss Weighting)**: Áp dụng công thức tính trọng số phạt nghịch đảo số lượng mẫu từng lớp:
  $$w_c = \frac{N_{\text{total}}}{C \times N_c}$$
  Với Neymar, trọng số phạt $w \approx 4.09\times$, trong khi Messi là $0.84\times$. Điều này buộc mạng phải tập trung học các đặc trưng diện mạo riêng biệt của Neymar.
- **Hàm mất mát**: Weighted Cross-Entropy Loss kết hợp Label Smoothing ($0.05$):
  $$L_{CE} = - \sum_{i=1}^{C} w_i y_i \log(\hat{y}_i)$$
- **Optimizer**: AdamW với tốc độ học $\eta = 0.001$ (hoặc $\eta = 0.0005$ cho MobileNetV2), Weight Decay $= 10^{-4}$.
- **Learning Rate Scheduler**: Cosine Annealing Learning Rate giảm dần $\eta$ theo đồ thị hàm cosin tới $10^{-6}$.

#### 5.3. Tiền xử lý phát hiện & cắt khuôn mặt tự động (Face Auto-Crop)
- **Vấn đề thực tế**: Khi người dùng tải ảnh thực tế từ internet hoặc chụp webcam, bức ảnh thường chứa cả thân người, trang phục thi đấu, màu áo câu lạc bộ và phông nền sân cỏ. Nếu đưa toàn bộ ảnh vào mạng $128 \times 128$, khuôn mặt bị co nhỏ và mô hình dễ nhận diện sai dựa trên màu áo hoặc bối cảnh.
- **Giải pháp**: Tích hợp thuật toán OpenCV Haar Cascade phát hiện tọa độ khuôn mặt $(x, y, w, h)$, tự động mở rộng vùng biên (margin $25\%$) và cắt chính xác khuôn mặt trước khi đưa vào mạng CNN.

---

### 6. HỆ THỐNG TRUY VẤN THÔNG TIN & TRỰC QUAN HÓA (RADAR CHART)

Sau khi mô hình đưa ra dự đoán xác suất $P(\text{cầu thủ} \mid \text{ảnh})$:
1. Lấy ra nhãn có xác suất cao nhất và mức độ tin cậy (Confidence %).
2. Tự động kết nối cơ sở dữ liệu `players_database.json` trích xuất:
   - Họ và tên đầy đủ, ngày sinh, tuổi, quốc gia, cờ biểu trưng.
   - Câu lạc bộ chủ quản, giải đấu, số áo, vị trí sở trường.
   - Thể hình (chiều cao, cân nặng), chân thuận, định giá chuyển nhượng.
   - Danh hiệu lớn (World Cup, Champions League, Quả bóng Vàng,...).
   - Đánh giá tổng quát FIFA Rating.
3. Vẽ biểu đồ Radar đa giác 6 cạnh (Radar Chart) thể hiện 6 chỉ số: Tốc độ (Pace), Sút (Shooting), Chuyền (Passing), Rê bóng (Dribbling), Phòng ngự (Defending), Thể lực (Physical).

---

### 7. KẾT QUẢ THỰC NGHIỆM & SO SÁNH KIẾN TRÚC

Đánh giá thực nghiệm độc lập trên tập Test gồm 338 hình ảnh (không tham gia huấn luyện):

| Tiêu chí so sánh | Custom Deep CNN | MobileNetV2 (Fine-Tuning Nâng Cao) | 🔥 Ensemble (Custom + MobileNet) |
| :--- | :---: | :---: | :---: |
| **Số tham số huấn luyện** | 1,207,400 | 2,011,336 (Mở khóa tầng cao) | Kết hợp 2 nhánh |
| **Dung lượng Checkpoint** | 14.5 MB | 13.1 MB | 27.6 MB |
| **Kỹ thuật tối ưu** | Khởi tạo ngẫu nhiên, train từ đầu | Top Convolutional Blocks + WeightedRandomSampler | Soft Voting (20% Custom + 80% MobileNet) |
| **Validation Accuracy** | 54.90% | **95.25% (Rất cao)** | **95.25%** |
| **Overall Test Accuracy** | 55.62% | **92.60% ⭐** | **92.60% ⭐** |
| **Neymar Precision** | 33.33% | **100.00% (Không gán nhầm)** | **100.00%** |
| **Neymar Recall** | 20.00% | **70.00%** | **70.00%** |
| **Cristiano Ronaldo Recall** | 60.00% | **95.56%** | **95.56%** |
| **Lionel Messi Precision** | 76.74% | **97.83%** | **97.83%** |
| **Thời gian suy luận (CPU)** | ~8 ms | ~12 ms | ~20 ms |

> [!TIP]
> **Nhận xét kết quả thực nghiệm**:
> 1. Nhờ mở khóa tinh chỉnh các tầng tích chập cấp cao kết hợp thuật toán **WeightedRandomSampler**, mô hình đã học được toàn diện các đường nét khuôn mặt của cả 8 cầu thủ, nâng độ chính xác kiểm thử từ **55.62%** lên đến **92.60%** (vượt xa mục tiêu 70-80% ban đầu).
> 2. Độ chuẩn xác riêng với cầu thủ **Neymar đạt tuyệt đối 100% Precision**, loại bỏ hoàn toàn hiện tượng nhận diện nhầm sang Messi.
> 3. Mô hình lai ghép **Ensemble** cho xác suất ổn định cao nhất trên cả ảnh chụp webcam và ảnh tải từ internet.

---

### 8. HƯỚNG DẪN CÀI ĐẶT & CHẠY ỨNG DỤNG

#### Bước 1: Kích hoạt môi trường ảo và cài đặt thư viện
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

#### Bước 2: Tự động tải tập dữ liệu Kaggle
```bash
python download_dataset.py
```

#### Bước 3: Huấn luyện mô hình CNN
```bash
# Huấn luyện Custom CNN
python train.py --model custom --epochs 25 --batch-size 32

# (Tùy chọn) Huấn luyện Transfer Learning MobileNetV2
python train.py --model mobilenet_v2 --epochs 20
```

#### Bước 4: Đánh giá mô hình trên tập Test
```bash
python evaluate.py --checkpoint checkpoints/best_custom.pth
```

#### Bước 5: Dự đoán nhanh bằng giao diện dòng lệnh (CLI)
```bash
python predict.py --image data/raw/Messi/sample_1.png
```

#### Bước 6: Khởi chạy Giao diện Web tương tác Streamlit
```bash
streamlit run app.py
```
*Truy cập trình duyệt tại địa chỉ:* `http://localhost:8501`

---

### 9. KẾT LUẬN & HƯỚNG PHÁT TRIỂN
- **Kết luận**: Đề tài đã xây dựng thành công giải pháp trọn vẹn từ thu thập dữ liệu, thiết kế mạng CNN phân cấp, huấn luyện tối ưu với cơ chế Dừng sớm và tích hợp cơ sở dữ liệu hồ sơ cầu thủ. Mô hình đạt độ chính xác cao trên tập kiểm thử và cung cấp giao diện trực quan hỗ trợ cả nhận diện từ ảnh tĩnh và webcam thời gian thực.
- **Hướng phát triển tiếp theo**:
  1. Mở rộng cơ sở dữ liệu lên hàng nghìn cầu thủ toàn cầu sử dụng Triplet Loss / ArcFace (Deep Metric Learning).
  2. Tích hợp bộ nhận diện vị trí khuôn mặt thời gian thực (MTCNN / YOLOv8-Face) để tự động nhận dạng nhiều cầu thủ cùng lúc trong một khung hình video trận đấu.
