# BÁO CÁO CHI TIẾT BÀI TẬP LỚN
## HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN (MULTIMEDIA DBMS)

**Đề tài:** Xây dựng Hệ thống Lưu trữ và Tìm kiếm Ảnh Lá Cây Tương đồng dựa trên Nội dung (Content-Based Leaf Image Retrieval - Leaf CBIR)  
**Tập trung giải quyết:** Yêu cầu 1 (Thu thập & Thống kê Bộ Dữ Liệu) và Yêu cầu 2 (Xây dựng Bộ Đặc trưng Nhận diện Tương đồng & Khác biệt)

---

## MỤC LỤC
1. [PHẦN 1: THU THẬP VÀ MÔ TẢ ĐẶC TÍNH BỘ DỮ LIỆU ẢNH LÁ CÂY](#phần-1-thu-thập-và-mô-tả-đặc-tính-bộ-dữ-liệu-ảnh-lá-cây)
   - 1.1. Tiêu chuẩn và Quy cách Thu thập Bộ Dữ liệu
   - 1.2. Thống kê Định lượng Bộ Dữ liệu (Quantitative Statistics)
   - 1.3. Phân tích Dưới góc nhìn Cảm nhận Thị giác (Visual & Perceptual Analysis)
   - 1.4. Đối chiếu So sánh Trực quan Side-by-Side giữa các Cặp Loài Đặc trưng
2. [PHẦN 2: XÂY DỰNG CÁC BỘ ĐẶC TRƯNG NHẬN DIỆN LÁ CÂY (FEATURE ENGINEERING)](#phần-2-xây-dựng-các-bộ-đặc-trưng-nhận-diện-lá-cây-feature-engineering)
   - 2.1. Kiến trúc Tiền xử lý & Phân đoạn Viền Sắc nét bằng AI (BiRefNet_lite)
   - 2.2. Không gian Đặc trưng Truyền thống 25 Chiều (25D Handcrafted Feature Vector)
     - 2.2.1. Nhóm Đặc trưng Màu sắc (Color Descriptors - 6D)
     - 2.2.2. Nhóm Đặc trưng Hình dạng Hình học (Shape Descriptors - 11D)
     - 2.2.3. Nhóm Đặc trưng Kết cấu Gân lá (GLCM Texture Descriptors - 8D)
   - 2.3. Chuẩn hóa Không gian Vector (Z-Score Normalization)
   - 2.4. Hàm Đo Khoảng cách Trọng số và Độ Tương đồng Phần trăm
   - 2.5. Đối chứng Nâng cao: Handcrafted Features vs Deep Learning Features (ResNet-18)
3. [KẾT LUẬN VÀ GIÁ TRỊ THỰC TIỄN](#phần-3-kết-luận-và-giá-trị-thực-tiễn)

---

# PHẦN 1: THU THẬP VÀ MÔ TẢ ĐẶC TÍNH BỘ DỮ LIỆU ẢNH LÁ CÂY

## 1.1. Tiêu chuẩn và Quy cách Thu thập Bộ Dữ liệu

Theo yêu cầu chuẩn hóa trong các hệ cơ sở dữ liệu đa phương tiện (MMDBMS), để việc trích xuất đặc trưng và tính toán khoảng cách không bị ảnh hưởng bởi các yếu tố nhiễu ngoại cảnh (nhiễu góc nghiêng phối cảnh, nhiễu tỉ lệ phóng đại, nhiễu nền phức tạp), bộ dữ liệu ảnh lá cây được xây dựng và tuyển chọn khắt khe theo các tiêu chí sau:

1. **Số lượng ảnh thu thập:**
   - Đạt tổng cộng **1240 ảnh** (vượt chỉ tiêu tối thiểu 1000 ảnh).
   - Chia làm hai tập tách biệt:
     - Tập Cơ sở dữ liệu lưu trữ (Gallery Database): **1200 ảnh** (8 loài $\times$ 150 ảnh/loài).
     - Tập Ảnh truy vấn kiểm thử độc lập (Test Query Set): **40 ảnh** (8 loài $\times$ 5 ảnh/loài).
2. **Quy cách chụp ảnh (Top-down Vertical View):**
   - Tất cả các mẫu lá cây đều được đặt phẳng trên mặt phẳng chụp.
   - Ống kính máy ảnh đặt vuông góc $90^\circ$ từ trên nhìn thẳng đứng xuống phiến lá, loại bỏ triệt để hiện tượng méo góc phối cảnh (perspective distortion).
   - Mỗi ảnh **chỉ chứa duy nhất một phiến lá cây đại diện**, được đặt cân đối ở trung tâm ảnh.
3. **Độ phân giải và Tỉ lệ khung hình (Aspect Ratio & Resolution):**
   - Tất cả **1240 ảnh** được chuẩn hóa đồng nhất về kích thước: **$416 \times 416$ pixels** (tỉ lệ khung hình vuông $1:1$).
   - Định dạng ảnh: **JPEG (24-bit RGB, 3 kênh màu)**, tương thích tối đa với các bộ giải mã ảnh chuẩn trong OpenCV, Pillow và trình duyệt Web.
   - Vật thể lá cây trong ảnh chiếm tỉ lệ diện tích từ $40\% - 75\%$ toàn bộ khung hình, đảm bảo tính đồng dạng về quy mô (scale similarity) giữa các ảnh.

```
       Khung ảnh chuẩn 416 x 416 pixels (JPEG / RGB)
  +------------------------------------------------------+
  |                   Nền tương phản                     |
  |                                                      |
  |                      ^                               |
  |                     / \     Đỉnh lá (Apex)           |
  |                    /   \                             |
  |                   /     \                            |
  |                  |   |   |  Phiến lá (Blade)         |
  |                  | --+-- |  Hệ gân lá (Veins)        |
  |                   \  |  /                            |
  |                    \ | /    Gốc lá (Base)            |
  |                      |      Cuống lá (Petiole)       |
  |                                                      |
  |            Chụp thẳng đứng 90 độ từ trên xuống       |
  +------------------------------------------------------+
```

---

## 1.2. Thống kê Định lượng Bộ Dữ liệu (Quantitative Statistics)

Bộ dữ liệu bao gồm **8 loài cây nông nghiệp và cây ăn quả phổ biến**, đại diện cho các hình thái phiến lá và đặc tính sinh học phong phú:

| STT | Tên Loài Cây (Tiếng Anh) | Tên Tiếng Việt | Tên Khoa Học | Số lượng ảnh CSDL | Số lượng ảnh Test | Tỉ lệ trong tập dữ liệu |
|:---:|:---|:---|:---|:---:|:---:|:---:|
| 1 | **Apple** | Cây Táo tây | *Malus domestica* | 150 | 5 | 12.5% |
| 2 | **Cherry** | Cây Anh đào | *Prunus avium* | 150 | 5 | 12.5% |
| 3 | **Grape** | Cây Nho | *Vitis vinifera* | 150 | 5 | 12.5% |
| 4 | **Peach** | Cây Đào | *Prunus persica* | 150 | 5 | 12.5% |
| 5 | **Pepper** | Cây Ớt chuông | *Capsicum annuum* | 150 | 5 | 12.5% |
| 6 | **Potato** | Cây Khoai tây | *Solanum tuberosum* | 150 | 5 | 12.5% |
| 7 | **Strawberry**| Cây Dâu tây | *Fragaria $\times$ ananassa* | 150 | 5 | 12.5% |
| 8 | **Tomato** | Cây Cà chua | *Solanum lycopersicum* | 150 | 5 | 12.5% |
| **Tổng** | **8 loài** | | | **1200** | **40** | **100%** |

> **Nhận xét về tính cân bằng dữ liệu (Class Balance):**  
> Bộ dữ liệu đạt tính cân bằng tuyệt đối ($150$ mẫu mỗi loài). Điều này triệt tiêu hoàn toàn hiện tượng thiên lệch mẫu (class bias), giúp các phép đo khoảng cách trong CSDL và các thuật toán học máy (K-Means, PCA, ResNet) hội tụ công bằng, không bị thống trị bởi loài có số lượng vượt trội.

### Bảng Thống Kê Các Chỉ Số Sinh Học Đo Lường Được (Trích xuất từ 1200 ảnh)

Dưới đây là giá trị trung bình ($\mu$) và độ lệch chuẩn ($\sigma$) của các thông số hình thái học thực tế được trích xuất trên toàn bộ 1200 ảnh:

| Loài cây | Sắc độ Hue $\mu_H$ (0-180) | Bão hòa Sat $\mu_S$ (0-255) | Độ sáng Val $\mu_V$ (0-255) | Độ tròn Circularity (0-1) | Độ lồi Solidity (0-1) | Khung Aspect Ratio ($W/H$) | Tỉ lệ Extent (0-1) | Độ tương phản GLCM Contrast |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Apple** | $59.1 \pm 6.8$ | $90.7 \pm 18.2$ | $104.4 \pm 21.0$ | $0.58 \pm 0.08$ | $0.90 \pm 0.03$ | $0.88 \pm 0.12$ | $0.59 \pm 0.05$ | $4.03 \pm 1.45$ |
| **Cherry** | $48.0 \pm 5.9$ | $112.0 \pm 20.4$ | $124.7 \pm 19.8$ | $0.56 \pm 0.07$ | $0.90 \pm 0.02$ | $0.92 \pm 0.10$ | $0.54 \pm 0.04$ | $3.06 \pm 1.12$ |
| **Grape** | $45.3 \pm 7.1$ | $132.5 \pm 22.1$ | $108.3 \pm 18.5$ | $0.45 \pm 0.06$ | $0.85 \pm 0.04$ | $0.95 \pm 0.09$ | $0.64 \pm 0.05$ | $5.68 \pm 1.80$ |
| **Peach** | $39.2 \pm 8.4$ | $141.5 \pm 25.3$ | $119.7 \pm 22.4$ | $0.44 \pm 0.05$ | $0.87 \pm 0.03$ | $0.86 \pm 0.14$ | $0.45 \pm 0.06$ | $2.89 \pm 0.95$ |
| **Pepper** | $50.8 \pm 6.2$ | $114.7 \pm 19.5$ | $114.9 \pm 17.6$ | $0.66 \pm 0.06$ | $0.92 \pm 0.02$ | $0.91 \pm 0.11$ | $0.60 \pm 0.05$ | $5.62 \pm 1.63$ |
| **Potato** | $41.5 \pm 7.6$ | $101.9 \pm 23.0$ | $120.7 \pm 20.1$ | $0.66 \pm 0.07$ | $0.95 \pm 0.02$ | $0.96 \pm 0.08$ | $0.65 \pm 0.04$ | $6.22 \pm 1.94$ |
| **Strawberry** | $43.4 \pm 6.5$ | $126.0 \pm 21.8$ | $133.9 \pm 24.0$ | $0.61 \pm 0.08$ | $0.93 \pm 0.03$ | $0.97 \pm 0.09$ | $0.69 \pm 0.05$ | $6.72 \pm 2.10$ |
| **Tomato** | $46.9 \pm 7.3$ | $116.6 \pm 20.9$ | $109.8 \pm 19.2$ | $0.47 \pm 0.07$ | $0.85 \pm 0.05$ | $0.85 \pm 0.13$ | $0.56 \pm 0.06$ | $4.81 \pm 1.72$ |

---

## 1.3. Phân tích Dưới góc nhìn Cảm nhận Thị giác (Visual & Perceptual Analysis)

Khi con người (chuyên gia hoặc người bình thường) quan sát bằng mắt thường các ảnh lá cây trong bộ dữ liệu, các thuộc tính thị giác được xử lý thông qua vỏ não thị giác bao gồm: màu sắc, đường viền hình học bao quanh, hoa văn kết cấu bề mặt và độ gợn sóng của mép lá.

### A. Những Điểm Tương Đồng Chung (Similarities)
1. **Bố cục không gian trung tâm:** Tất cả mẫu vật đều là một lá đơn (hoặc cụm lá chét gắn liền) nằm ở vị trí trung tâm, xung quanh là phông nền chụp có độ tương phản cao so với phiến lá.
2. **Tính đối xứng sinh học trục (Bilateral Symmetry):** Đại đa số các lá cây đều phát triển theo quy luật đối xứng song song qua một đường gân chính trung tâm (Primary midrib vein) kéo dài từ cuống lá đến đỉnh lá.
3. **Màu sắc nền tảng:** Sắc xanh lục chiếm ưu thế tuyệt đối do chất diệp lục ($Chlorophyll\ a$ và $b$). Phổ màu chủ đạo tập trung ở dải bước sóng xanh lá cây (Hue từ $35^\circ$ đến $65^\circ$ trong không gian HSV).

### B. Những Điểm Khác Biệt Sâu Sắc (Differences)
1. **Hình thái tổng thể của phiến lá (Blade Morphology):**
   - *Dạng thuôn mác (Lanceolate):* Lá Đào (**Peach**) có thân lá rất dài, thon nhỏ dần về hai đầu, chiều rộng phiến lá rất hẹp so với chiều dài.
   - *Dạng hình trứng rộng / Bầu dục (Ovate / Elliptical):* Lá Ớt chuông (**Pepper**), Khoai tây (**Potato**), Táo (**Apple**) có thân phình to ở giữa hoặc phần dưới gần cuống, bo tròn mềm mại.
   - *Dạng xẻ thùy sâu (Lobed Blade):* Lá Nho (**Grape**) và lá Cà chua (**Tomato**) không có đường viền liền mạch mà bị khía lõm sâu tạo thành các thùy (lobes) xòe rộng.
2. **Đặc điểm đường mép lá (Leaf Margin):**
   - *Mép nguyên (Entire margin):* Lá Ớt chuông (**Pepper**) và lá Đào (**Peach**) có đường biên trơn tru, không có răng cưa.
   - *Mép răng cưa thô (Coarsely Serrated / Dentate):* Lá Dâu tây (**Strawberry**) có các khía răng cưa tam giác rất to, đều đặn như lưỡi cưa gỗ.
   - *Mép răng cưa mịn (Serrulate / Crenate):* Lá Táo (**Apple**) và Anh đào (**Cherry**) có răng cưa rất nhỏ và nông, nhìn từ xa gần như trơn láng.
3. **Cấu trúc mạng gân lá (Venation Architecture):**
   - *Gân hình lông chim (Pinnate venation):* Lá Táo, Đào, Anh đào, Ớt chuông có một gân chính nổi rõ ở giữa, các gân phụ tủa đều sang hai bên như xương cá.
   - *Gân hình chân vịt (Palmate venation):* Lá Nho có nhiều gân chính lớn cùng xuất phát từ một điểm ở cuống lá, tủa tròn ra các đỉnh thùy như bàn tay xòe.
4. **Độ bóng bề mặt và Kết cấu vi mô (Surface Texture & Gloss):**
   - Lá Anh đào (**Cherry**) có lớp cutin sáp mỏng trên bề mặt, tạo cảm giác bóng loáng, nhẵn nhụi, ít vân vi mô.
   - Lá Dâu tây (**Strawberry**) và Khoai tây (**Potato**) có bề mặt thô nhám, gân lá phụ chằng chịt nổi gồ ghề, tạo nên độ tương phản cục bộ rất cao khi ánh sáng rọi vào.

---

## 1.4. Đối chiếu So sánh Trực quan Side-by-Side giữa các Cặp Loài Đặc trưng

Để làm rõ khả năng phân biệt của các giác quan và kiểm chứng cơ sở dữ liệu, chúng tôi thực hiện so sánh đối đầu từng cặp loài điển hình:

```
+------------------------------------------------------------------------------------------------------------------+
|                                    BẢNG ĐỐI CHIẾU SIDE-BY-SIDE 4 CẶP LOÀI ĐẶC TRƯNG                             |
+------------------------------------------------------------------------------------------------------------------+
| CẶP SO SÁNH               | ĐIỂM TƯƠNG ĐỒNG BẰNG MẮT THƯỜNG               | ĐIỂM KHÁC BIỆT BẰNG MẮT THƯỜNG               |
+---------------------------+-----------------------------------------------+----------------------------------------------+
| 1. Peach (Đào)            | - Đều là lá đơn, mép viền trơn láng.          | - Peach: Dáng thuôn nhọn như mũi mác hẹp     |
|    vs                     | - Gân lá hình lông chim đối xứng.             |   (Extent chỉ 0.45, Circularity 0.44).       |
|    Pepper (Ớt chuông)     | - Sắc tố xanh lục tương đối đồng đều.         | - Pepper: Phiến lá hình bầu dục tròn trịa    |
|                           |                                               |   (Extent đạt 0.60, Circularity 0.66).       |
+---------------------------+-----------------------------------------------+----------------------------------------------+
| 2. Grape (Nho)            | - Đều có đường viền xẻ thùy sâu (lobed).      | - Grape: Gân chân vịt xòe bàn tay, phiến lá  |
|    vs                     | - Độ lồi Solidity thấp (~0.85).               |   rộng bè ngang, gốc lá hình tim (cordate).  |
|    Tomato (Cà chua)       | - Mép lá gợn sóng ngoằn ngoèo.                | - Tomato: Gân lông chim, các thùy lá xẻ bất  |
|                           |                                               |   đối xứng và thân lá thuôn dài hơn.         |
+---------------------------+-----------------------------------------------+----------------------------------------------+
| 3. Strawberry (Dâu tây)   | - Khung hình chữ nhật vuông vức (AR ~ 0.9).   | - Strawberry: Răng cưa rất to, gân lá phụ    |
|    vs                     | - Độ lồi Solidity cao (>0.90).                |   nổi cộm rõ rệt (GLCM Contrast 6.72).       |
|    Apple (Táo tây)        | - Phiến lá rộng, bo tròn đầy đặn.             | - Apple: Mép viền chỉ hơi gợn răng cưa mịn,   |
|                           |                                               |   bề mặt mịn màng hơn (GLCM Contrast 4.03).  |
+---------------------------+-----------------------------------------------+----------------------------------------------+
| 4. Cherry (Anh đào)       | - Đều có tỉ lệ kích thước cân đối.            | - Cherry: Bề mặt lá nhẵn bóng, trơn láng,    |
|    vs                     | - Hình dáng elip bo tròn ở giữa thân.         |   độ đồng nhất cao (GLCM Homogeneity 0.63).  |
|    Potato (Khoai tây)     | - Màu xanh tự nhiên của cây thân thảo/gỗ.     | - Potato: Bề mặt sần sùi thô ráp, gân phụ    |
|                           |                                               |   chìm nổi rõ (GLCM Homogeneity 0.52).       |
+------------------------------------------------------------------------------------------------------------------+
```

*(Minh họa đồ họa trực quan chi tiết xem tại file ảnh đính kèm: `visualization/11_dataset_leaf_species_grid.png` và `visualization/12_side_by_side_visual_comparison.png`)*.

---

# PHẦN 2: XÂY DỰNG CÁC BỘ ĐẶC TRƯNG NHẬN DIỆN LÁ CÂY (FEATURE ENGINEERING)

Trong hệ thống truy vấn ảnh theo nội dung (CBIR), ảnh không thể được so sánh từng điểm ảnh (pixel-by-pixel) vì kích thước quá lớn, độ phức tạp $O(W \times H)$ cao và nhạy cảm với ánh sáng, tịnh tiến. Do đó, cần ánh xạ mỗi ảnh từ không gian điểm ảnh không có cấu trúc sang một vector đặc trưng toán học cô đọng:

$$\mathbf{x} \in \mathbb{R}^D \quad (D \ll W \times H)$$

Hệ thống của chúng tôi xây dựng một **bộ đặc trưng kết hợp 25 chiều (25D Handcrafted Feature Vector)** bao quát toàn diện 3 khía cạnh thị giác then chốt: **Màu sắc (Color)**, **Hình dạng (Shape)** và **Kết cấu gân lá (Texture)**.

```
                      VECTOR ĐẶC TRƯNG 25 CHIỀU (25D FEATURE VECTOR)
 +-------------------------+-----------------------------------+-----------------------------+
 |     COLOR (6 Dims)      |          SHAPE (11 Dims)          |      TEXTURE (8 Dims)       |
 | [0..5]: H, S, V         | [6..12]: 7 Hu Moments             | [17..22]: 6 GLCM Properties |
 |         (Mean & Std)    | [13..16]: AR, Circ, Solid, Extent | [23..24]: Gray Mean & Std   |
 +-------------------------+-----------------------------------+-----------------------------+
```

---

## 2.1. Kiến trúc Tiền xử lý & Phân đoạn Viền Sắc nét bằng AI (BiRefNet_lite)

Để trích xuất được hình dạng và kết cấu chuẩn xác, bước tiên quyết là phải **tách hoàn toàn chiếc lá ra khỏi nền ảnh** (Segmentation).
- Các phương pháp truyền thống (như ngưỡng Otsu hay lọc màu sắc) thường bị thất bại khi gặp bóng đổ của cuống lá, gân lá bị thủng lỗ hoặc viền lá bị đứt quãng.
- Chúng tôi tích hợp mô hình AI phân đoạn SOTA mang tên **BiRefNet_lite** (Bilateral Reference Network for High-Resolution Dichotomous Image Segmentation):
  - Mô hình tiếp nhận ảnh đầu vào $1024 \times 1024$ pixels, sử dụng kiến trúc hai nhánh (Bilateral) bảo tồn đồng thời thông tin ngữ cảnh toàn cục và chi tiết vi mô cạnh sắc nét.
  - Đầu ra là mặt nạ nhị phân xác suất $P(x, y) \in [0, 1]$. Áp dụng ngưỡng cắt $\theta = 0.5$ (tức pixel $> 128$) để thu được mặt nạ nhị phân hoàn hảo $\mathbf{M} \in \{0, 255\}^{H \times W}$.
  - Từ mặt nạ $\mathbf{M}$, sử dụng thuật toán dò biên Suzuki & Abe (`cv2.findContours`) để trích xuất đường viền khép kín lớn nhất của lá $\mathcal{C} = \{(x_i, y_i)\}_{i=1}^N$.

---

## 2.2. Không gian Đặc trưng Truyền thống 25 Chiều (25D Handcrafted Feature Vector)

### 2.2.1. Nhóm Đặc trưng Màu sắc (Color Descriptors - 6 Chiều)

#### Lý thuyết & Cơ sở Toán học
Không gian màu RGB tuy trực quan trên màn hình nhưng có sự tương quan chặt chẽ giữa 3 kênh và phụ thuộc mạnh vào cường độ chiếu sáng. Do đó, chúng tôi chuyển đổi ảnh sang **không gian màu HSV (Hue - Saturation - Value)**:
- **Kênh H (Hue - Sắc thái màu):** Thể hiện màu sắc thuần khiết độc lập với độ sáng (từ $0^\circ$ đến $360^\circ$, OpenCV chuẩn hóa về $[0, 180]$).
- **Kênh S (Saturation - Độ bão hòa):** Thể hiện độ thuần khiết của màu (từ nhạt/xám đến đậm nét, $[0, 255]$).
- **Kênh V (Value - Giá trị độ sáng):** Thể hiện mức độ phản xạ ánh sáng của phiến lá ($[0, 255]$).

Chỉ xét trên tập hợp các pixel thuộc vùng lá cây $\Omega_{leaf} = \{(x, y) \mid \mathbf{M}(x, y) = 255\}$, số lượng pixel là $N_{leaf} = |\Omega_{leaf}|$.

#### Công thức Giải tích
Hai mô-men màu (Color Moments) cấp 1 và cấp 2 được tính trên từng kênh $c \in \{H, S, V\}$:

1. **Giá trị trung bình (Mean - Mô-men cấp 1):**
   $$\mu_c = \frac{1}{N_{leaf}} \sum_{(x, y) \in \Omega_{leaf}} I_c(x, y)$$
2. **Độ lệch chuẩn (Standard Deviation - Mô-men cấp 2):**
   $$\sigma_c = \sqrt{\frac{1}{N_{leaf}} \sum_{(x, y) \in \Omega_{leaf}} \Big(I_c(x, y) - \mu_c\Big)^2}$$

Vector màu sắc 6 chiều:
$$\mathbf{f}_{color} = \Big[ \mu_H, \sigma_H, \mu_S, \sigma_S, \mu_V, \sigma_V \Big]^T$$

#### Giá trị Thông tin & Ý nghĩa Ứng dụng
- $\mu_H$ xác định sắc thái xanh lá cây (xanh ngả vàng như lá Đào $\mu_H \approx 39.2$ hay xanh đậm như lá Táo $\mu_H \approx 59.1$).
- $\sigma_H$ đo độ loang màu hoặc sự không đồng đều của diệp lục.
- $\mu_S, \sigma_S$ phân biệt độ đậm đà của sắc tố sinh học.
- $\mu_V, \sigma_V$ phản ánh mức độ sáng/tối tự nhiên của từng loài lá.

---

### 2.2.2. Nhóm Đặc trưng Hình dạng Hình học (Shape Descriptors - 11 Chiều)

Hình dạng là đặc trưng quan trọng nhất để nhận dạng loài thực vật vì nó mang tính di truyền cao và ít bị biến đổi bởi môi trường dinh dưỡng so với màu sắc.

#### A. 7 Bất biến Hu Moments ($\phi_1 \dots \phi_7$ - 7 Chiều)
Hu Moments là tập hợp 7 phương trình mô-men phi tuyến nổi tiếng được Ming-Kuei Hu đề xuất năm 1962, sở hữu tính chất toán học kỳ diệu: **Bất biến hoàn toàn đối với phép tịnh tiến (Translation), co giãn (Scale) và phép quay (Rotation)**.

1. **Mô-men hình học không gian cấp $(p+q)$:**
   $$m_{pq} = \sum_{(x, y) \in \Omega_{leaf}} x^p y^q$$
   Tâm của lá cây: $\bar{x} = \frac{m_{10}}{m_{00}}, \quad \bar{y} = \frac{m_{01}}{m_{00}}$
2. **Mô-men trung tâm (Central Moments - Bất biến tịnh tiến):**
   $$\mu_{pq} = \sum_{(x, y) \in \Omega_{leaf}} (x - \bar{x})^p (y - \bar{y})^q$$
3. **Mô-men trung tâm chuẩn hóa (Normalized Central Moments - Bất biến co giãn):**
   $$\eta_{pq} = \frac{\mu_{pq}}{\mu_{00}^{\gamma}}, \quad \text{với } \gamma = \frac{p+q}{2} + 1 \quad (\forall p+q \ge 2)$$
4. **7 Bất biến Hu (Hu Invariant Moments - Bất biến phép quay):**
   $$\begin{aligned}
   \phi_1 &= \eta_{20} + \eta_{02} \\
   \phi_2 &= (\eta_{20} - \eta_{02})^2 + 4\eta_{11}^2 \\
   \phi_3 &= (\eta_{30} - 3\eta_{12})^2 + (3\eta_{21} - \eta_{03})^2 \\
   \phi_4 &= (\eta_{30} + \eta_{12})^2 + (\eta_{21} + \eta_{03})^2 \\
   \phi_5 &= (\eta_{30} - 3\eta_{12})(\eta_{30} + \eta_{12})\Big[(\eta_{30} + \eta_{12})^2 - 3(\eta_{21} + \eta_{03})^2\Big] \\
          &\quad + (3\eta_{21} - \eta_{03})(\eta_{21} + \eta_{03})\Big[3(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2\Big] \\
   \phi_6 &= (\eta_{20} - \eta_{02})\Big[(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2\Big] + 4\eta_{11}(\eta_{30} + \eta_{12})(\eta_{21} + \eta_{03}) \\
   \phi_7 &= (3\eta_{21} - \eta_{03})(\eta_{30} + \eta_{12})\Big[(\eta_{30} + \eta_{12})^2 - 3(\eta_{21} + \eta_{03})^2\Big] \\
          &\quad - (\eta_{30} - 3\eta_{12})(\eta_{21} + \eta_{03})\Big[3(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2\Big]
   \end{aligned}$$

*Kỹ thuật ổn định số học (Log-transformation):*  
Do giá trị của $\phi_k$ dao động ở các lũy thừa rất nhỏ ($10^{-3}$ đến $10^{-20}$), chúng tôi áp dụng phép biến đổi logarit cơ số 10 có dấu để đưa về dải đo tương thích:
$$h_k = -\text{sign}(\phi_k) \cdot \log_{10}(|\phi_k| + 10^{-12})$$

#### B. 4 Thuộc tính Hình học Sinh học Cốt lõi (Biometric Descriptors - 4 Chiều)
Gọi $A$ là diện tích phiến lá (Contour Area), $P$ là chu vi viền lá (Perimeter / Arc Length), $\mathcal{H}$ là bao lồi (Convex Hull) với diện tích $A_{hull}$, và $[W_{box}, H_{box}]$ là kích thước hộp chữ nhật bao nhỏ nhất (Bounding Bounding Box).

1. **Tỉ lệ khung hình (Aspect Ratio):**
   $$AR = \frac{W_{box}}{H_{box}}$$
   *Ý nghĩa:* Nhận dạng phiến lá thon dài hẹp ($AR$ nhỏ) hay tròn dẹt bè ngang ($AR \approx 1$).
2. **Độ tròn / Độ kết đặc (Circularity / Compactness):**
   $$C = \frac{4\pi A}{P^2} \quad (0 < C \le 1)$$
   *Ý nghĩa:* Một hình tròn lý tưởng có $C = 1$. Lá cây càng có viền răng cưa sâu, mép uốn lượn ngoằn ngoèo hoặc xẻ thùy thì chu vi $P$ càng lớn, kéo theo $C$ giảm mạnh (ví dụ lá Nho $C \approx 0.45$, lá Ớt tròn $C \approx 0.66$).
3. **Độ lồi / Độ đặc chắc (Solidity):**
   $$S = \frac{A}{A_{hull}} \quad (0 < S \le 1)$$
   *Ý nghĩa:* Phát hiện mức độ xẻ thùy lõm của phiến lá. Lá có phiến lá nguyên liền mạch như Khoai tây, Ớt đạt $S \approx 0.92 - 0.95$. Ngược lại, lá Nho và Cà chua xẻ thùy tạo nhiều khoảng trống lõm vào trong bao lồi, dẫn đến $S$ tụt xuống $\approx 0.85$.
4. **Tỉ lệ chiếm hộp (Extent):**
   $$E = \frac{A}{W_{box} \cdot H_{box}} \quad (0 < E \le 1)$$
   *Ý nghĩa:* Phân biệt mức độ chiếm dụng không gian của phiến lá so với hộp chữ nhật bao bọc.

Vector hình dạng 11 chiều:
$$\mathbf{f}_{shape} = \Big[ h_1, h_2, h_3, h_4, h_5, h_6, h_7, AR, C, S, E \Big]^T$$

---

### 2.2.3. Nhóm Đặc trưng Kết cấu Gân lá (GLCM Texture Descriptors - 8 Chiều)

Kết cấu (Texture) phản ánh sự phân bố không gian mức xám trên bề mặt lá, đặc biệt là hệ thống gân lá chính phụ và độ nhẵn nhụi của lớp biểu bì. Chúng tôi sử dụng phương pháp kinh điển và hiệu quả nhất trong xử lý ảnh y sinh và thị giác: **Ma trận Đồng mức xám GLCM (Gray-Level Co-occurrence Matrix)** do Robert Haralick phát minh.

#### Nguyên lý Tính Toán GLCM
1. Ảnh mức xám được lượng tử hóa từ 256 mức xuống **$N_g = 32$ mức xám** ($I_{gray} \gets \lfloor I / 8 \rfloor$). Thao tác này giúp nén không gian ma trận xuống kích thước nhỏ $32 \times 32$, khử nhiễu lượng tử hóa và tăng tốc độ xử lý hơn 64 lần.
2. Với khoảng cách không gian pixel $d \in \{1, 3\}$ và 4 góc hướng $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$, phần tử $P(i, j \mid d, \theta)$ là xác suất để hai pixel cách nhau khoảng cách $d$ theo hướng $\theta$ có mức xám lần lượt là $i$ và $j$.
3. Ma trận sau khi tính được đối xứng hóa và chuẩn hóa tổng xác suất bằng 1: $\sum_{i,j} P(i, j) = 1$. Lấy trung bình cộng ma trận trên cả 4 hướng để đảm bảo **tính bất biến với hướng xoay**.

#### 6 Chỉ số Thống kê Haralick + 2 Chỉ số Phân bố Mức xám:
1. **Độ tương phản vi mô (Contrast):**
   $$\text{Contrast} = \sum_{i=0}^{N_g-1} \sum_{j=0}^{N_g-1} |i - j|^2 P(i, j)$$
   *Ý nghĩa:* Đo lường cường độ chênh lệch mức xám cục bộ giữa các pixel lân cận. Lá có gân nổi rõ, thô ráp (như lá Dâu tây: 6.72) có Contrast rất cao; lá nhẵn bóng (như lá Đào: 2.89) có Contrast rất thấp.
2. **Độ dị biệt tuyến tính (Dissimilarity):**
   $$\text{Dissimilarity} = \sum_{i=0}^{N_g-1} \sum_{j=0}^{N_g-1} |i - j| P(i, j)$$
3. **Độ đồng nhất cục bộ (Homogeneity / IDM):**
   $$\text{Homogeneity} = \sum_{i=0}^{N_g-1} \sum_{j=0}^{N_g-1} \frac{P(i, j)}{1 + (i - j)^2}$$
   *Ý nghĩa:* Giá trị cao khi các cặp pixel lân cận có mức xám gần bằng nhau (đường chéo chính của GLCM tập trung nhiều xác suất). Lá trơn bóng, đồng nhất (Cherry: 0.63) có Homogeneity cao.
4. **Năng lượng / Độ trật tự (Energy / Angular Second Moment - ASM):**
   $$\text{Energy} = \sqrt{\text{ASM}} = \sqrt{\sum_{i=0}^{N_g-1} \sum_{j=0}^{N_g-1} P(i, j)^2}$$
   *Ý nghĩa:* Đo tính lặp lại trật tự của cấu trúc bề mặt.
5. **Mức độ tương quan không gian (Correlation):**
   $$\text{Correlation} = \sum_{i=0}^{N_g-1} \sum_{j=0}^{N_g-1} \frac{(i - \mu_i)(j - \mu_j) P(i, j)}{\sigma_i \sigma_j}$$
   *Ý nghĩa:* Thể hiện sự liên kết tuyến tính giữa các dải vân lá liền kề.
6. **Mức xám trung bình & Độ lệch chuẩn mức xám ($\mu_{gray}, \sigma_{gray}$):**
   Đo độ hấp thụ ánh sáng trung bình và độ tán xạ mức xám của bề mặt biểu bì phiến lá.

Vector kết cấu 8 chiều:
$$\mathbf{f}_{texture} = \Big[ \text{Contrast}, \text{Dissimilarity}, \text{Homogeneity}, \text{Energy}, \text{Correlation}, \text{ASM}, \mu_{gray}, \sigma_{gray} \Big]^T$$

---

## 2.3. Chuẩn hóa Không gian Vector (Z-Score Normalization)

Trong hệ thống vector 25 chiều $\mathbf{f} \in \mathbb{R}^{25}$, các thuộc tính có đơn vị và thang đo vật lý hoàn toàn khác biệt:
- Sắc độ Hue $\mu_H \in [0, 180]$.
- Độ tròn Circularity $C \in [0, 1]$.
- Hu moments sau log $h_k \in [0, 30]$.
- GLCM Contrast $\in [1, 15]$.

Nếu đo khoảng cách trực tiếp, thuộc tính có biên độ lớn ($\mu_H$) sẽ áp đảo hoàn toàn các thuộc tính có biên độ nhỏ ($C, S$), dẫn đến sai lệch nghiêm trọng. Do đó, toàn bộ ma trận đặc trưng trong CSDL được chuẩn hóa theo phương pháp **Z-Score Normalization**:

$$z_{i, d} = \frac{f_{i, d} - \mu_d}{\sigma_d}$$

Trong đó $\mu_d$ và $\sigma_d$ là giá trị trung bình và độ lệch chuẩn của chiều thuộc tính thứ $d$ ($d = 1 \dots 25$) tính trên toàn bộ 1200 ảnh CSDL. Sau khi chuẩn hóa, mỗi chiều đặc trưng đều có $\mu = 0$ và $\sigma = 1$, đưa mọi thuộc tính về một hệ quy chiếu công bằng tuyệt đối.

---

## 2.4. Hàm Đo Khoảng cách Trọng số và Độ Tương đồng Phần trăm

Khi người dùng đưa một ảnh truy vấn $\mathcal{Q}$ vào hệ thống:
1. Hệ thống trích xuất vector 25 chiều $\mathbf{f}_q$, sau đó chuẩn hóa bằng tham số $(\mu_d, \sigma_d)$ của CSDL thu được $\mathbf{z}_q$.
2. Khoảng cách thành phần giữa ảnh truy vấn $\mathbf{z}_q$ và ảnh thứ $i$ trong CSDL $\mathbf{z}_i$ được tính riêng cho từng nhóm:
   - **Khoảng cách Hình dạng (11 chiều):**
     $$D_{shape}(q, i) = \frac{1}{\sqrt{11}} \sqrt{ \sum_{d=6}^{16} (z_{q, d} - z_{i, d})^2 }$$
   - **Khoảng cách Kết cấu gân lá (8 chiều):**
     $$D_{texture}(q, i) = \frac{1}{\sqrt{8}} \sqrt{ \sum_{d=17}^{24} (z_{q, d} - z_{i, d})^2 }$$
   - **Khoảng cách Màu sắc (6 chiều):**
     $$D_{color}(q, i) = \frac{1}{\sqrt{6}} \sqrt{ \sum_{d=0}^{5} (z_{q, d} - z_{i, d})^2 }$$
3. **Khoảng cách Tổng hợp có Trọng số (Weighted Total Distance):**
   $$D_{total}(q, i) = w_{shape} D_{shape}(q, i) + w_{texture} D_{texture}(q, i) + w_{color} D_{color}(q, i)$$
   *(Với $w_{shape} + w_{texture} + w_{color} = 1.0$. Mặc định: $w_{shape} = 0.50$, $w_{texture} = 0.30$, $w_{color} = 0.20$)*.
4. **Quy đổi sang Độ Tương Đồng Phần Trăm (Similarity Percentage):**
   Khoảng cách Euclidean $D \in [0, +\infty)$. Để hiển thị trực quan và thân thiện với người dùng, hệ thống chuyển đổi sang thang đo $[0\%, 100\%]$ bằng hàm phân rã mũ (Exponential Decay):
   $$\text{Similarity}(q, i) = 100 \cdot \exp\left( -\frac{D_{total}(q, i)}{\tau} \right) \quad (\text{với } \tau = 1.5)$$
   - Khi hai lá giống nhau hoàn hảo: $D_{total} = 0 \implies \text{Similarity} = 100\%$.
   - Khi độ lệch tăng dần, độ tương đồng suy giảm mượt mà về $0\%$.

---

## 2.5. Đối chứng Nâng cao: Handcrafted Features vs Deep Learning Features (ResNet-18)

Bên cạnh bộ đặc trưng truyền thống 25 chiều, chúng tôi đã triển khai mở rộng 2 cấp độ học sâu (Deep CNN) để làm đối chứng khoa học toàn diện:
1. **Cấp độ 2 - Pretrained ResNet-18 (512D):** Sử dụng mạng nơ-ron tích chập ResNet-18 đã được tiền huấn luyện trên ImageNet làm bộ trích xuất vector đặc trưng vạn năng (*Zero-shot Transfer Learning*).
2. **Cấp độ 3 - Fine-tuned ResNet-18 (512D):** Huấn luyện tinh chỉnh mạng nơ-ron trực tiếp trên tập lá cây 8 loài bằng hàm mất mát Cross-Entropy Loss trong 15 epoch trên GPU NVIDIA RTX 4060, đạt độ chính xác kiểm định **99.58%**.

### Bảng Kết Quả Thực Nghiệm Đối Chứng 3 Cấp Độ (Đo trên 40 Test Queries & 1200 Ảnh CSDL)

| Chỉ số đánh giá | Ý nghĩa khoa học | Cấp 1: Handcrafted 25D (CBIR) | Cấp 2: Pretrained ResNet-18 (512D) | Cấp 3: Fine-tuned ResNet-18 (512D) |
|:---|:---|:---:|:---:|:---:|
| **Precision@1 (Top-1)** | Tỉ lệ ảnh xếp hạng 1 đúng loài cây | **52.5%** | **87.5%** | **100.0%** 🚀 |
| **Precision@3 (Top-3)** | Tỉ lệ ảnh trong Top 3 đúng loài cây | **53.3%** | **83.3%** | **100.0%** 🚀 |
| **Precision@5 (Top-5)** | Tỉ lệ ảnh trong Top 5 đúng loài cây | **50.0%** | **82.0%** | **100.0%** 🚀 |
| **mAP@5** | Độ chính xác trung bình (Mean Average Precision) | **62.9%** | **91.0%** | **100.0%** 🚀 |
| **Cluster Purity (K-Means)** | Độ thuần khiết nhãn trong cụm ($k=8$) | **36.8%** | **59.2%** | **99.9%** 🚀 |
| **Adjusted Rand Index (ARI)** | Mức độ trùng khớp với loài sinh học `[0, 1]` | **0.1272** | **0.3613** | **0.9981** 🚀 |
| **Silhouette Score** | Độ gắn kết nội cụm và cách biệt ngoại cụm | **0.1043** | **0.0726** | **0.4853** 🚀 |
| **Tính giải thích (Explainability)**| Khả năng con người đọc hiểu và diễn giải | **Rất cao (Hiểu rõ độ tròn, gân, màu)** | Thấp (Vector tiềm ẩn trừu tượng) | Thấp (Vector tiềm ẩn trừu tượng) |
| **Tính linh hoạt người dùng** | Cho phép chỉnh slider tùy biến trọng số | **Có (Tùy biến 100% Shape / Color / Texture)** | Không (Vector cố định) | Không (Vector cố định) |

---

# PHẦN 3: KẾT LUẬN VÀ GIÁ TRỊ THỰC TIỄN

1. **Về Yêu cầu 1 (Bộ dữ liệu):**
   - Đã thu thập và chuẩn hóa thành công **1240 ảnh lá cây chất lượng cao**, phân bố đồng đều hoàn hảo trên 8 loài thực vật quan trọng.
   - Các ảnh thỏa mãn tuyệt đối tiêu chuẩn ảnh đa phương tiện: chụp thẳng đứng $90^\circ$ từ trên xuống, một lá duy nhất ở trung tâm, độ phân giải đồng nhất $416 \times 416$ pixels, định dạng JPEG.
   - Đã phân tích thấu đáo các đặc tính giống và khác nhau cả về phương diện định lượng sinh học lẫn cảm nhận thị giác của con người, có bảng đối chiếu side-by-side chi tiết.
2. **Về Yêu cầu 2 (Hệ thống đặc trưng nhận diện):**
   - Đã thiết kế bộ đặc trưng **25D Handcrafted** hoàn chỉnh, vững chắc về mặt toán học: kết hợp trọn vẹn Màu sắc (HSV moments), Hình dạng bất biến (Hu moments, Circularity, Solidity, Aspect Ratio, Extent) và Kết cấu gân lá (GLCM Haralick).
   - Nêu rõ công thức giải tích, ý nghĩa vật lý và giá trị thông tin của từng thuộc tính trong việc tìm kiếm sự tương đồng và phân tách sự khác biệt.
   - Xây dựng thành công cơ chế chuẩn hóa Z-Score và hàm khoảng cách có trọng số, cho phép người dùng tương tác tùy chỉnh tiêu chí truy vấn trên giao diện Web.
   - Mở rộng đối chứng khoa học với phương pháp Học sâu (Deep CNN ResNet-18), nâng độ chính xác tìm kiếm từ $50\%$ lên **$100\%$**, tạo nên một công trình nghiên cứu hoàn chỉnh, chặt chẽ và đạt chất lượng học thuật cao nhất cho môn học **Cơ Sở Dữ Liệu Đa Phương Tiện**.

---
*Báo cáo được hoàn thiện và xuất bản từ hệ thống Leaf CBIR System - Khoa Công nghệ Thông tin.*
