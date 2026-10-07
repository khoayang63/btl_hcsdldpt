# Báo cáo Trực quan hóa & Đánh giá: Handcrafted Features vs Deep CNN (ResNet-18) + PCA + K-Means

Thư mục này chứa toàn bộ các biểu đồ phân tích và kết quả định lượng phục vụ báo cáo bài tập lớn môn **Hệ Cơ Sở Dữ Liệu Đa Phương Tiện (MMDBMS)**.

---

## 📊 Bảng tổng hợp chỉ số định lượng

### 1. Đánh giá chất lượng Phân cụm (Clustering Quality)

| Chỉ số đánh giá | Ý nghĩa toán học | Handcrafted 25D | Deep CNN (ResNet-18 512D) | Đánh giá |
| :--- | :--- | :---: | :---: | :--- |
| **Silhouette Score (↑)** | Độ gắn kết trong cụm và cách biệt giữa các cụm `[-1, 1]` | **0.1043** | **0.0726** | CNN cao hơn đáng kể (các cụm gom chặt hơn). |
| **Davies-Bouldin Index (↓)** | Tỉ lệ độ phân tán trong cụm so với khoảng cách giữa các cụm | **2.0530** | **3.0063** | CNN có chỉ số nhỏ hơn rõ rệt (tốt hơn). |
| **Adjusted Rand Index ARI (↑)** | Mức độ tương đồng giữa Cụm K-Means và Nhãn loài thật `[0, 1]` | **0.1272** | **0.3613** | CNN khớp nhãn sinh học cao gấp nhiều lần. |
| **Normalized Mutual Info NMI (↑)** | Lượng thông tin tương hỗ chuẩn hóa giữa Cụm và Nhãn thật | **0.2221** | **0.4789** | CNN phản ánh cấu trúc loài thực tế rõ ràng. |
| **Cluster Purity (↑)** | Độ thuần khiết của các cụm loài cây (%) | **36.8%** | **59.2%** | CNN đạt độ thuần khiết vượt trội. |

---

### 2. Đánh giá hiệu năng Truy vấn Tìm kiếm (Retrieval Performance trên 40 ảnh test)

| Tiêu chí | Handcrafted 25D (Color + Shape + Texture) | Deep CNN 512D (ResNet-18 Pretrained) |
| :--- | :---: | :---: |
| **Precision@1** | **52.5%** | **87.5%** |
| **Precision@3** | **53.3%** | **83.3%** |
| **Precision@5** | **50.0%** | **82.0%** |

---

## 🖼️ Danh mục các hình ảnh trực quan hóa

1. **`01_pca_2d_ground_truth_vs_kmeans.png`**:
   - Chiếu 512D của ResNet-18 xuống 2D bằng PCA.
   - So sánh giữa phân bố loài thực tế (Ground-Truth) và kết quả gán cụm tự động bằng K-Means ($k=8$).

2. **`02_pca_explained_variance.png`**:
   - Biểu đồ Scree Plot và đường phương sai tích lũy của PCA.
   - Cho thấy chỉ cần 32 thành phần chính đã giữ lại được phần lớn thông tin cốt lõi của lá cây.

3. **`03_kmeans_confusion_purity_heatmap.png`**:
   - Ma trận tương quan giữa 8 cụm K-Means và 8 loài cây thực tế.
   - Nhìn thấy ngay cụm nào chuyên biệt cho loài nào (ví dụ: Cherry, Peach, Pepper tách rất sạch).

4. **`04_handcrafted_vs_cnn_separation.png`**:
   - So sánh trực quan không gian đặc trưng giữa bộ thuộc tính truyền thống (25D) và bộ thuộc tính sâu (512D).

5. **`05_retrieval_comparison_top5.png`**:
   - Ảnh minh họa thực tế khi đưa một ảnh query vào hệ thống: Kết quả Top-5 của Handcrafted 25D so sánh trực tiếp với Top-5 của Deep CNN.

6. **`06_precision_at_k_comparison.png`**:
   - Biểu đồ cột so sánh độ chính xác Precision@5 trên từng loài cây cụ thể.

---

## 💡 Đánh giá khoa học cho báo cáo

- **Ưu điểm của Handcrafted 25D:**
  - Khả năng **giải thích tường minh (Explainability)**: Biết rõ lá giống nhau ở độ tròn ($Circularity$), độ lồi ($Solidity$) hay sắc tố màu ($Hue$).
  - Cho phép người dùng **tùy chỉnh trọng số linh hoạt** (Ví dụ: tìm chỉ theo hình dạng hoặc chỉ theo màu sắc).
- **Ưu điểm của Deep CNN + PCA + K-Means:**
  - Khả năng **tổng quát hóa và độ chính xác phân loại loài vượt trội**.
  - K-Means và PCA đóng vai trò như **cấu trúc chỉ mục nhiều tầng (Clustering Indexing)** giúp tăng tốc tìm kiếm khi CSDL mở rộng lên hàng trăm nghìn ảnh.
