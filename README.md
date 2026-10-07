# Hệ Thống Lưu Trữ và Tìm Kiếm Ảnh Lá Cây Tương Đồng (Leaf CBIR)

> **Báo cáo & Đồ án môn học:** Cơ Sở Dữ Liệu Đa Phương Tiện (MMDBMS)  
> **Phương pháp:** Trích xuất đặc trưng đa chiều (25D Feature Vector: Color + Shape + Texture) kết hợp mô hình AI phân đoạn viền sắc nét **BiRefNet_lite**.

---

## 📖 Giới thiệu hệ thống

Hệ thống cho phép quản trị CSDL ảnh lá cây và tìm kiếm các ảnh có độ tương đồng cao nhất (**Top-5 Most Similar**) dựa trên nội dung ảnh (Content-Based Image Retrieval - CBIR):
1. **Phân đoạn lá cây chính xác cao:** Tích hợp mô hình AI **BiRefNet_lite** (SOTA Image Segmentation) loại bỏ hoàn toàn bóng đổ, tách trọn cuống lá và viền răng cưa.
2. **Bộ đặc trưng 25 chiều (25D Feature Vector):**
   - **Màu sắc (Color - 6D):** Trung bình và độ lệch chuẩn của 3 kênh HSV ($H_{mean}, H_{std}, S_{mean}, S_{std}, V_{mean}, V_{std}$).
   - **Hình dạng (Shape - 11D):** 7 bất biến Hu Moments (biến đổi log) + Tỉ lệ khung hình (Aspect Ratio), Độ tròn (Circularity), Độ lồi/xẻ thùy (Solidity), Extent.
   - **Kết cấu gân lá (Texture - 8D):** 6 chỉ số ma trận đồng mức xám GLCM (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM) + Thống kê mức xám (Gray Mean, Gray Std).
3. **Bộ máy tìm kiếm & Tiêu chí linh hoạt (Search Engine):**
   - Chuẩn hóa Z-Score trên toàn bộ CSDL.
   - Khoảng cách Euclid có trọng số, chuyển đổi sang phần trăm tương đồng theo hàm phân rã mũ ($Similarity = 100 \cdot e^{-D / 1.5}$).
   - Cho phép chọn nhanh các chế độ tìm kiếm: **Tổng hợp**, **Chỉ Hình dạng (100%)**, **Chỉ Màu sắc (100%)**, **Chỉ Gân lá (100%)**, **Hình dạng & Gân lá (bỏ qua màu)** hoặc **Tùy chỉnh Slider**.
4. **Trực quan hóa chi tiết & So sánh 2 lá (Side-by-Side Comparison):**
   - Bảng so sánh 25 chỉ số chi tiết giữa Query và Match.
   - Biểu đồ Radar Profile đối chiếu trực quan.
   - Khung hình học chồng lớp (Contour, Convex Hull, Bounding Box, Fitted Ellipse).

---

## 📂 Cấu trúc thư mục

```
hcsdlpt/
├── app.py                      # Server web demo (Flask)
├── extract_features.py         # Script trích xuất 25D vector & nạp CSDL SQLite
├── extract_cnn_and_evaluate.py # Script trích xuất ResNet-18 512D (Pretrained), PCA 2D & K-Means
├── train_leaf_model.py         # [CẤP 3] Huấn luyện Fine-tuned ResNet-18 (Validation Acc: 99.58%)
├── benchmark_and_3d_pca.py     # [BENCHMARK] Đối chứng toàn diện 3 cấp độ & sinh PCA 3D
├── setup_dataset.py            # Script giải nén & phân loại dataset từ zip
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── .gitignore                  # Loại bỏ file nặng/môi trường khỏi git
│
├── src/                        # Mã nguồn chính của hệ thống
│   ├── database.py             # Quản trị SQLite (bảng categories, images, features)
│   ├── feature_extractor.py    # Tích hợp BiRefNet_lite + Trích xuất 25D handcrafted feature
│   ├── cnn_extractor.py        # Trích xuất Deep CNN Embeddings (ResNet-18 512D L2-norm)
│   └── search_engine.py        # Thuật toán tìm kiếm Top-5 (Handcrafted, Pretrained & Fine-tuned)
│
├── visualization/              # Biểu đồ trực quan hóa & báo cáo đánh giá khoa học
│   ├── 01_pca_2d_ground_truth_vs_kmeans.png
│   ├── 02_pca_explained_variance.png
│   ├── 03_kmeans_confusion_purity_heatmap.png
│   ├── 04_handcrafted_vs_cnn_separation.png
│   ├── 05_retrieval_comparison_top5.png
│   ├── 06_precision_at_k_comparison.png
│   ├── 07_training_curves.png              # [MỚI] Đường cong huấn luyện Loss & Accuracy 15 epoch
│   ├── 08_benchmark_3_levels_comparison.png# [MỚI] So sánh định lượng 3 Cấp độ
│   ├── 09_pca_3d_view.png                  # [MỚI] Không gian đặc trưng PCA 3D (Pretrained vs Fine-tuned)
│   ├── 09_pca_3d_interactive.html         # [MỚI] Trang HTML tương tác 3D xoay 360 độ (Plotly)
│   ├── 10_confusion_matrix_finetuned.png   # [MỚI] Ma trận nhầm lẫn phân loại Validation (99.58%)
│   └── README_VISUALIZATION.md             # Báo cáo chi tiết các chỉ số định lượng
│
├── data/                       # Dữ liệu ảnh & metadata
│   ├── dataset/                # 1200 ảnh trong CSDL (8 loài x 150 ảnh)
│   ├── test_queries/           # 40 ảnh query mẫu (8 loài x 5 ảnh)
│   ├── dataset_metadata.json
│   └── test_queries_metadata.json
│
├── db/                         # Cơ sở dữ liệu & Cache
│   ├── leaf_database.db        # SQLite Database
│   ├── feature_cache.npz       # Ma trận vector 25D chuẩn hóa
│   ├── cnn_feature_cache.npz   # Ma trận vector 512D Pretrained CNN & PCA 2D
│   └── finetuned_feature_cache.npz # [MỚI] Ma trận vector 512D Fine-tuned CNN
│
├── templates/                  # Giao diện web HTML (Flask template)
│   └── index.html
└── static/                     # Tài nguyên web tĩnh & upload tạm thời
```

---

## 🛠️ Hướng dẫn cài đặt & Chạy chương trình

### 1. Yêu cầu môi trường
- **Hệ điều hành:** Windows / Linux / macOS
- **Python:** Khuyến nghị Python 3.10 đến 3.13
- **GPU (tùy chọn):** NVIDIA GPU với CUDA để mô hình BiRefNet chạy nhanh nhất (khoảng 30ms/ảnh). Nếu không có GPU, hệ thống tự động chạy trên CPU hoặc fallback về thuật toán Otsu.

---

### 2. Tạo môi trường ảo (venv) & Cài đặt thư viện

Mở terminal tại thư mục gốc của dự án:

```bash
# 1. Tạo môi trường ảo venv độc lập
python -m venv venv

# 2. Kích hoạt môi trường ảo
# Trên Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Hoặc trên Windows (Command Prompt):
.\venv\Scripts\activate.bat
# Trên Linux/macOS:
source venv/bin/activate
```

#### Cài đặt PyTorch (với hỗ trợ GPU CUDA):
Nếu máy có card đồ họa NVIDIA (khuyên dùng):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Nếu chỉ chạy trên CPU:
```bash
pip install torch torchvision
```

#### Cài đặt các thư viện còn lại từ `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

### 3. Chuẩn bị dữ liệu & Cơ sở dữ liệu (Nếu chạy từ đầu)

Nếu bạn vừa clone repository về và chưa có dữ liệu giải nén:
```bash
# 1. Giải nén ảnh từ tập tin gốc LEAF.v2i.coco.zip vào data/
python setup_dataset.py

# 2. Trích xuất đặc trưng bằng BiRefNet và nạp CSDL SQLite + tạo file cache
python extract_features.py
```

---

### 4. Khởi chạy Web Demo

```bash
python app.py
```

Sau khi terminal hiển thị thông báo server hoạt động, mở trình duyệt web và truy cập:
👉 **`http://localhost:5000`** (hoặc `http://127.0.0.1:5000`)

---

## 🎯 Hướng dẫn sử dụng giao diện
1. **Chọn ảnh truy vấn:**
   - Bấm vào khung **"Click to upload"** để tải lên ảnh lá bất kỳ từ máy tính.
   - Hoặc click vào các thẻ tên trong phần **"Test Samples"** để thử nghiệm nhanh các ảnh mẫu chuẩn.
2. **Chọn tiêu chí truy vấn (Dropdown):**
   - *🌿 Tổng hợp Handcrafted (Hình dạng 50% + Gân 30% + Màu 20%)*
   - *🧠 Deep CNN Embedding (ResNet-18 512D) [Độ chính xác cao 87.5%]*
   - *📐 Chỉ theo Hình dạng (100% Shape)*
   - *🎨 Chỉ theo Màu sắc (100% Color)*
   - *🧬 Chỉ theo Gân lá (100% Texture)*
   - *📐+🧬 Hình dạng & Gân lá (Không xét màu sắc)*
   - *⚙️ Tùy chỉnh Slider theo tỷ lệ mong muốn*
3. **Tìm kiếm & Quan sát:**
   - Nhấn **"Search Similar Leaves"**.
   - Xem kết quả **Top 5 ảnh giống nhất** kèm điểm % tương đồng.
4. **Đối chiếu & So sánh chi tiết:**
   - Nhấp chuột trực tiếp vào bất kỳ ảnh nào trong Top 5 để mở **Modal So sánh Chi tiết (Side-by-Side)**: so khớp hình học, phân tích độ lệch từng chỉ số và biểu đồ Radar Profile.
5. **Đánh giá Phân cụm & Giảm chiều:**
   - Chuyển sang Tab **"🧠 K-Means & PCA"** để xem trực tiếp các biểu đồ phân cụm, Scree plot, Confusion Matrix heatmap và biểu đồ đối sánh Precision@5 giữa 2 phương pháp.

---

## 📊 Bảng so sánh định lượng: Đối chứng 3 Cấp độ

Chạy script đánh giá độc lập:
```bash
# Huấn luyện mô hình Cấp 3 (Fine-tuning 15 epoch trên GPU):
python train_leaf_model.py

# Đánh giá đối chứng toàn diện 3 cấp độ & sinh biểu đồ + trang 3D tương tác:
python benchmark_and_3d_pca.py
```

### 1. Đánh giá chất lượng Phân cụm (Clustering Quality - K-Means $k=8$)

| Chỉ số đánh giá | Ý nghĩa toán học | Cấp 1: Handcrafted 25D | Cấp 2: Pretrained 512D | Cấp 3: Fine-tuned 512D | Đánh giá học thuật |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Cluster Purity (↑)** | Độ thuần khiết nhãn loài trong từng cụm (%) | **36.8%** | **59.2%** | **99.9%** | Fine-tuned đạt độ thuần khiết gần như tuyệt đối (+63.1%). |
| **Adjusted Rand Index ARI (↑)** | Tương đồng giữa Cụm K-Means và Nhãn loài thật `[0, 1]` | **0.1272** | **0.3613** | **0.9981** | Cấu trúc cụm khớp nhãn sinh học thực tế gần như 1.0 tuyệt đối. |
| **Normalized Mutual Info NMI (↑)**| Lượng thông tin tương hỗ chuẩn hóa giữa Cụm và Nhãn thật | **0.2221** | **0.4789** | **0.9976** | Phản ánh cấu trúc loài cây trồng thực tế rõ ràng vượt bậc. |
| **Silhouette Score (↑)** | Độ cô đặc nội cụm và tách biệt giữa các cụm `[-1, 1]` | **0.1043** | **0.0726** | **0.4853** | Tăng vọt gấp 5 lần, các cụm cô đặc cực kỳ chặt chẽ. |
| **Davies-Bouldin Index (↓)** | Tỉ lệ độ phân tán trong cụm so với khoảng cách cụm | **2.0530** | **3.0063** | **0.8629** | Chỉ số nhỏ nhất thể hiện các cụm cách ly rõ rệt nhất. |

### 2. Đánh giá hiệu năng Truy vấn Tìm kiếm (Retrieval Performance trên 40 test queries)

| Tiêu chí | Cấp 1: Handcrafted 25D | Cấp 2: Pretrained ResNet-18 | Cấp 3: Fine-tuned ResNet-18 | Cải thiện Cấp 3 vs Cấp 1 |
| :--- | :---: | :---: | :---: | :---: |
| **Precision@1 (Top-1)** | **52.5%** | **87.5%** | **100.0%** | **+47.5%** 🚀 |
| **Precision@3 (Top-3)** | **53.3%** | **83.3%** | **100.0%** | **+46.7%** 🚀 |
| **Precision@5 (Top-5)** | **50.0%** | **82.0%** | **100.0%** | **+50.0%** 🚀 |
| **mAP@5 (Mean Average Precision)** | **62.9%** | **91.0%** | **100.0%** | **+37.1%** 🚀 |

---

## 🌌 Trực quan hóa Không gian Vector Đặc trưng PCA 3D

- **Hình ảnh tĩnh 3D:** [`visualization/09_pca_3d_view.png`](visualization/09_pca_3d_view.png) so sánh trực quan không gian đặc trưng 3D giữa Pretrained (tản mác) và Fine-tuned (gom thành 8 đảo độc lập).
- **Trang web 3D tương tác xoay 360 độ:** Mở file [`visualization/09_pca_3d_interactive.html`](visualization/09_pca_3d_interactive.html) bằng bất kỳ trình duyệt nào để xoay tự do, phóng to thu nhỏ và rê chuột xem tọa độ từng chiếc lá.
- Toàn bộ báo cáo phân tích chi tiết nằm trong [`visualization/README_VISUALIZATION.md`](visualization/README_VISUALIZATION.md).
