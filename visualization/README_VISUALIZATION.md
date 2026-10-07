# Báo cáo Trực quan hóa & Đánh giá Đối chứng 3 Cấp độ
## Handcrafted 25D vs Pretrained ResNet-18 vs Fine-tuned ResNet-18

Báo cáo khoa học thuộc đề tài: **Hệ Cơ Sở Dữ Liệu Đa Phương Tiện (Multimedia DBMS) - CBIR Leaf Retrieval**.

---

## 📊 Bảng tổng hợp đối chứng định lượng (Benchmark Summary)

### 1. Đánh giá chất lượng Phân cụm (Clustering Quality - K-Means $k=8$)

| Chỉ số đánh giá | Ý nghĩa toán học | Cấp 1: Handcrafted 25D | Cấp 2: Pretrained 512D | Cấp 3: Fine-tuned 512D | Nhận xét học thuật |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Cluster Purity (↑)** | Độ thuần khiết nhãn loài trong từng cụm (%) | **36.8%** | **59.2%** | **99.9%** | Fine-tuned đạt độ thuần khiết gần như tuyệt đối (+60.5% so với Cấp 1). |
| **Adjusted Rand Index ARI (↑)** | Độ khớp giữa Cụm K-Means và Nhãn loài thật `[0, 1]` | **0.1272** | **0.3613** | **0.9981** | Khớp cấu trúc sinh học gấp gần 8 lần so với phương pháp truyền thống. |
| **Normalized Mutual Info NMI (↑)**| Lượng thông tin tương hỗ chuẩn hóa `[0, 1]` | **0.2221** | **0.4789** | **0.9976** | Phản ánh cấu trúc nhãn cây trồng thực tế rõ ràng vượt bậc. |
| **Silhouette Score (↑)** | Độ cô đặc nội cụm và tách biệt giữa các cụm | **0.1043** | **0.0726** | **0.4853** | Fine-tuned kéo chặt vector cùng loài lại với nhau trong không gian sâu. |
| **Davies-Bouldin Index (↓)** | Tỉ lệ độ phân tán trong cụm so với khoảng cách cụm | **2.0530** | **3.0063** | **0.8629** | Chỉ số nhỏ nhất thể hiện các cụm tách biệt rõ ràng nhất. |

---

### 2. Đánh giá hiệu năng Truy vấn Tìm kiếm (Retrieval Performance trên 40 test queries)

| Tiêu chí đánh giá | Cấp 1: Handcrafted 25D | Cấp 2: Pretrained ResNet-18 | Cấp 3: Fine-tuned ResNet-18 | Cải thiện Cấp 3 vs Cấp 1 |
| :--- | :---: | :---: | :---: | :---: |
| **Precision@1 (Top-1)** | **52.5%** | **87.5%** | **100.0%** | **+47.5%** 🚀 |
| **Precision@3 (Top-3)** | **53.3%** | **83.3%** | **100.0%** | **+46.7%** 🚀 |
| **Precision@5 (Top-5)** | **50.0%** | **82.0%** | **100.0%** | **+50.0%** 🚀 |
| **Mean Average Precision (mAP@5)** | **62.9%** | **91.0%** | **100.0%** | **+37.1%** 🚀 |

---

## 🖼️ Danh mục toàn bộ 10 File Trực quan hóa

1. **`01_pca_2d_ground_truth_vs_kmeans.png`**: Chiếu 2D PCA so sánh phân bố nhãn thật vs nhãn cụm K-Means.
2. **`02_pca_explained_variance.png`**: Scree plot và phương sai tích lũy (PCA 32D giải thích gần 70% phương sai).
3. **`03_kmeans_confusion_purity_heatmap.png`**: Ma trận phân bố các loài trong từng cụm K-Means.
4. **`04_handcrafted_vs_cnn_separation.png`**: So sánh trực quan không gian Handcrafted 25D vs CNN 512D.
5. **`05_retrieval_comparison_top5.png`**: Ảnh minh họa truy vấn thực tế Top-5: Handcrafted vs Deep CNN.
6. **`06_precision_at_k_comparison.png`**: Biểu đồ cột so sánh Precision@5 trên từng loài lá cây.
7. **`07_training_curves.png`**: **[MỚI]** Đường cong hàm mất mát (Loss) và độ chính xác (Accuracy) trong 15 epoch huấn luyện Fine-tuned ResNet-18.
8. **`08_benchmark_3_levels_comparison.png`**: **[MỚI]** Biểu đồ so sánh đối chứng toàn diện 3 cấp độ (Precision & Clustering Quality).
9. **`09_pca_3d_view.png` & `09_pca_3d_interactive.html`**: **[MỚI]** Không gian vector PCA 3D tương tác xoay 360 độ (Pretrained vs Fine-tuned).
10. **`10_confusion_matrix_finetuned.png`**: **[MỚI]** Ma trận nhầm lẫn phân loại trên tập Validation (Đạt 99.58% accuracy).

---

## 💡 Phân tích học thuật phục vụ bảo vệ đồ án

- **Cấp 1 (Handcrafted 25D):** Có tính giải thích cao (*Explainability*), người dùng có thể điều chỉnh trọng số theo ý muốn (chỉ tìm theo hình dáng hoặc chỉ tìm theo màu sắc), nhưng độ chính xác phân loại thấp do không nắm bắt được hoa văn phức tạp.
- **Cấp 2 (Pretrained ResNet-18):** Học chuyển giao (*Transfer Learning*) không cần nhãn (*Zero-shot Feature Extraction*), độ chính xác tăng vọt lên 87.5%, rất linh hoạt khi thêm loài mới (*Open-set*).
- **Cấp 3 (Fine-tuned ResNet-18):** Mạng nơ-ron được học chuyên sâu có giám sát (*Supervised Deep Metric Representation*), các vector cùng loài gom thành các đảo tách biệt độc lập trong không gian PCA 3D, đưa độ chính xác truy vấn lên mức hoàn hảo (>97% - 99%).
