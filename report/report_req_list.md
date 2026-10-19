# ĐỀ CƯƠNG CÁC ĐỀ MỤC BÁO CÁO CSDL ĐA PHƯƠNG TIỆN (YÊU CẦU 1 & 2)
> **Chủ đề:** Xây dựng Hệ Cơ sở dữ liệu và Trích xuất Đặc trưng Nhận diện Lá cây (Content-Based Leaf Image Retrieval - CBLIR)  
> **Tài liệu tham khảo cấu trúc:** `Báo cáo CSDL ĐPT yêu cầu 1 và 2.docx` (Ánh xạ mô hình từ đặc trưng âm thanh sang đặc trưng hình ảnh lá cây)  
> **Định dạng file draft:** `report/report_req_list.md` (Chỉ dùng để duyệt cấu trúc, chưa bao gồm toàn văn nội dung)

---

## 📌 TỔNG QUAN CẤU TRÚC BÁO CÁO

Báo cáo được thiết kế tương thích với quy chuẩn của học phần Cơ sở dữ liệu Đa phương tiện, ánh xạ từ mô hình xử lý tín hiệu đa phương tiện chuẩn sang miền thị giác máy tính (Computer Vision & Image Retrieval):

- **Trang bìa & Thông tin chung:** Tên trường, khoa, tên đề tài, giảng viên hướng dẫn, danh sách thành viên nhóm.
- **Lời nói đầu:** Đặt vấn đề về CSDL Đa phương tiện hình ảnh, tính cấp thiết của CBIR cho thực vật học nông nghiệp, mục tiêu giải quyết 2 yêu cầu nền tảng đầu tiên.
- **Mục lục:** Danh mục các mục, tiểu mục, danh mục hình ảnh và danh mục bảng biểu.
- **PHẦN I: XÂY DỰNG BỘ DỮ LIỆU ẢNH LÁ CÂY** (Giải quyết Yêu cầu 1 - Thu thập, chuẩn hóa, thống kê và đánh giá cân bằng dữ liệu).
- **PHẦN II: PHÂN TÍCH ĐẶC ĐIỂM GIỐNG VÀ KHÁC NHAU GIỮA CÁC LÁ CÂY** (Giải quyết Yêu cầu 1 - Cảm nhận thị giác, so sánh đối sánh side-by-side).
- **PHẦN III: XÂY DỰNG CÁC BỘ ĐẶC TRƯNG NHẬN DIỆN LÁ CÂY** (Giải quyết Yêu cầu 2 - Cơ sở toán học, công thức, giá trị thông tin của các nhóm: Color-based, Texture-based, Shape-based, Deep Learning, và thiết kế Vector tổng hợp).
- **PHẦN KẾT LUẬN VÀ HƯỚNG TRIỂN KHAI TIẾP THEO** (Chuyển tiếp sang Yêu cầu 3 & 4 về đánh chỉ mục và truy vấn Query by Example).

---

## 📑 DANH MỤC CÁC ĐỀ MỤC CHI TIẾT CẦN BÁO CÁO

### PHẦN MỞ ĐẦU
- [ ] **Trang bìa & Thông tin đề tài**
  - Tên đề tài: *Nghiên cứu, Xây dựng Hệ Cơ sở Dữ liệu và Tìm kiếm Ảnh Lá Cây Dựa Trên Nội Dung (Content-Based Leaf Image Retrieval - CBLIR)*.
  - Thông tin học phần, giảng viên, nhóm sinh viên thực hiện.
- [ ] **Mục lục, Danh mục Bảng, Danh mục Hình ảnh**
- [ ] **Lời nói đầu**
  - Bối cảnh bùng nổ dữ liệu đa phương tiện thị giác và sự chuyển dịch từ truy vấn text-based sang Content-Based Image Retrieval (CBIR).
  - Thách thức phân loại và tìm kiếm hình thái lá cây trong nông nghiệp số / thực vật học.
  - Mục tiêu báo cáo: Tập trung giải quyết 2 giai đoạn nền tảng: Xây dựng bộ dữ liệu ảnh chuẩn và Thiết kế các bộ vector đặc trưng nhận diện.

---

### PHẦN I. XÂY DỰNG BỘ DỮ LIỆU ẢNH LÁ CÂY (YÊU CẦU 1)

#### 1.1 Tổng quan & Nguồn dữ liệu
- [ ] **1.1.1. Mục tiêu và quy mô tập dữ liệu**
  - *Mục tiêu:* Phục vụ lưu trữ và truy vấn tương đồng hình ảnh lá cây bằng Query by Example (QbE).
  - *Quy mô:* Thỏa mãn chỉ tiêu tối thiểu $\ge 1.000$ files ảnh lá cây.
  - *Phân bổ các lớp loài:* Danh sách các loài lá cây thu thập (ví dụ: các loài cây lấy lá, cây ăn quả, cây thuốc hoặc tập lá chuẩn thực vật học).
  - *Tính đơn đối tượng:* Mỗi ảnh chỉ gồm đúng một lá cây duy nhất, góc chụp thẳng đứng trên xuống (top-down view), nền đồng nhất.
- [ ] **1.1.2. Nguồn dữ liệu và quy trình thu thập/sàng lọc**
  - Nguồn dữ liệu gốc (Kaggle Leaf Dataset / Flavia / Swedish Leaf / PlantVillage hoặc ảnh chụp thực nghiệm nhóm tự thu thập).
  - Tiêu chí sàng lọc loại bỏ ảnh nhiễu: loại bỏ ảnh mờ nhoè (motion blur), thiếu sáng, nhiều lá đè nhau, lá bị gãy cuống hoặc che khuất nghiêm trọng.

#### 1.2 Quy ước đặt tên, Cấu trúc lưu trữ & Mô hình hóa Đối tượng
- [ ] **1.2.1. Phân tích tên file gốc và lý do chuẩn hóa**
  - Phân tích hạn chế tên file gốc (tên quá dài, dấu cách, ký tự đặc biệt tiếng Việt / Unicode, rủi ro SQL injection / lỗi file path).
  - Yêu cầu đồng bộ định danh phục vụ đánh chỉ mục trong CSDL đa phương tiện.
- [ ] **1.2.2. Quy ước mã định danh (Leaf_ID) và quy tắc đặt tên chuẩn hóa**
  - Cấu trúc mã ID khóa chính: `[Tiền_tố_loài_2-3_ký_tự]_[Số_thứ_tự_4_chữ_số].jpg` (ví dụ: `LEAF_MANG_0001.jpg`, `LEAF_GUAV_0015.jpg`...).
  - Bảng mã định danh viết tắt cho toàn bộ các loài trong bộ dữ liệu.
- [ ] **1.2.3. Cấu trúc cây thư mục lưu trữ vật lý**
  - Cây thư mục tổ chức dữ liệu thô và dữ liệu đã qua tiền xử lý:
    - Thư mục gốc bộ dữ liệu (`dataset/`).
    - Các thư mục con chia theo từng loài/lớp.
    - Sơ đồ cấu trúc cây thư mục (dạng ASCII tree tương tự Hình 1 trong file mẫu).
- [ ] **1.2.4. Quản lý Metadata và mô hình hóa Đối tượng Đa phương tiện (Leaf Object Modeling)**
  - Tiếp cận Hướng đối tượng đa phương tiện (Object-Oriented Multimedia Representation):
    - *Thành phần dữ liệu thô (Raw Image Data Component):* Đường dẫn lưu trữ vật lý, định dạng ảnh, kích thước ảnh.
    - *Thành phần siêu dữ liệu ngữ nghĩa (Metadata / Ground Truth Attributes):* Loài cây, tên khoa học, họ thực vật, kiểu gân lá, dạng mép viền, nguồn thu thập.
    - *Thành phần vector đặc trưng nội dung (Content-based Feature Vector Attributes):* Mảng số đặc trưng hình thái, màu sắc, kết cấu và deep embeddings.
  - Bảng thiết kế lược đồ Metadata (Spreadsheet schema & các trường dữ liệu tương tự Hình 2 trong file mẫu).

#### 1.3 Thông số kỹ thuật số hóa & Tiêu chuẩn chuẩn hóa ảnh
- [ ] **1.3.1. Thông số kỹ thuật hình ảnh số hóa**
  - *Định dạng ảnh:* Chuẩn hóa JPEG/PNG (không nén mất mát chất lượng quá mức, tối ưu lưu trữ).
  - *Không gian màu:* RGB 3 kênh chuẩn (24-bit color depth), đồng thời chuyển đổi sang Grayscale và HSV khi xử lý đặc trưng.
  - *Độ phân giải & Kích thước cố định:* Chuẩn hóa toàn bộ ảnh về cùng độ phân giải (ví dụ: $256 \times 256$ pixels hoặc $512 \times 512$ pixels).
- [ ] **1.3.2. Căn cứ lựa chọn góc chụp thẳng đứng và tỷ lệ khung hình tương đồng**
  - Luận giải cơ sở khoa học của góc chụp thẳng đứng $90^\circ$ từ trên xuống (Top-down orthogonal projection):
    - Triệt tiêu biến dạng phối cảnh (Perspective Distortion).
    - Bảo toàn tỷ lệ diện tích thực và hình bao ngoài của phiến lá.
    - Cho phép trích xuất các bất biến hình học (Geometric Invariants) chuẩn xác.
  - Tiêu chuẩn tỷ lệ khung hình tương đồng (Aspect Ratio 1:1 hoặc giữ tỷ lệ thực kèm padding/bounding box):
    - Tránh hiện tượng méo giãn phi tuyến làm sai lệch hình dạng lá.
    - Phục vụ việc chia lưới (Grid partition) và trích xuất đặc trưng mạng nơ-ron CNN.

#### 1.4 Thống kê chi tiết các loài lá cây trong CSDL
- [ ] **1.4.1. Bảng tổng hợp số liệu các loài lá cây**
  - Bảng thống kê: Tên loài, Tên khoa học, Số lượng ảnh, Độ phân giải chuẩn, Tỷ lệ % trên toàn bộ CSDL.
- [ ] **1.4.2. Các biểu đồ phân bố thống kê của bộ dữ liệu**
  - *Biểu đồ 1:* Phân bố số lượng ảnh giữa các loài lá cây (đánh giá cân bằng quy mô).
  - *Biểu đồ 2:* Phân bố diện tích vùng lá (Leaf Area Distribution) và tỷ lệ che phủ khung hình (Extent).
  - *Biểu đồ 3:* Phân bố tỷ lệ co giãn khung bao (Aspect Ratio Distribution = Chiều dài / Chiều rộng phiến lá).
  - *Biểu đồ 4:* Phân bố giá trị màu sắc trung bình (Kênh màu H và G đại diện cho sắc thái xanh/vàng).
  - *Biểu đồ 5:* Phân bố tần suất màu sắc tổng hợp (Global RGB Histogram Distribution) trên toàn bộ CSDL.

#### 1.5 Đánh giá tính cân bằng và các hạn chế thực tế của bộ dữ liệu
- [ ] **1.5.1. Đánh giá tính cân bằng của bộ dữ liệu**
  - *a. Cân bằng cấp độ vĩ mô (Macro Balance - Phân bố số lượng mẫu giữa các loài):*
    - Tỷ lệ phân bổ mẫu giữa các loài; triệt tiêu hiện tượng thiên vị mật độ (Density Bias) trong không gian vector đa chiều khi truy vấn Top-K.
  - *b. Cân bằng về dạng hình thái học (Morphological Balance):*
    - Đối trọng giữa các nhóm hình thái lá: Lá hình tròn/bầu dục (Oval/Round) đối trọng với Lá thuôn dài (Lanceolate/Linear) và Lá xẻ thùy (Lobed).
    - Cân bằng giữa mép lá trơn nhẵn (Entire margin) và mép lá có răng cưa (Serrated/Dentate margin).
- [ ] **1.5.2. Các hạn chế tồn tại trong bộ dữ liệu thực tế**
  - *a. Biến thiên điều kiện chiếu sáng và bóng đổ (Illumination & Shadow Variations):*
    - Tác động đến đặc trưng màu sắc HSV/RGB và giải pháp chuẩn hóa độ sáng.
  - *b. Sai khác sinh trưởng nội loài (Intra-class Variation):*
    - Lá non (màu xanh non, kích thước nhỏ, gân mỏng) so với lá già (màu xanh thẫm, đốm vàng, gân nổi rõ) trong cùng một loài cây.
  - *c. Nhiễu khuyết tật tự nhiên (Natural Defects & Occlusions):*
    - Vết sâu cắn rách mép lá, đốm nấm bệnh làm thay đổi đường bao contour và các mô-men hình dạng.
  - *d. Giới hạn về góc xoay ngẫu nhiên (In-plane Rotation):*
    - Cuống lá quay theo nhiều hướng khác nhau trong ảnh đòi hỏi đặc trưng phải có tính bất biến quay (Rotation Invariant).

---

### PHẦN II. PHÂN TÍCH ĐẶC ĐIỂM GIỐNG VÀ KHÁC NHAU GIỮA CÁC LÁ CÂY (YÊU CẦU 1)

#### 2.1 Đặc điểm giống nhau giữa các loài lá cây (Kèm hình ảnh minh họa)
- [ ] **2.1.1. Cấu trúc sinh học và hình học tổng thể**
  - Đều có cấu tạo giải phẫu học cơ bản gồm 3 phần: Cuống lá (Petiole), Phiến lá (Blade/Lamina), và Hệ thống gân lá (Venation).
  - Đa số có tính đối xứng song phương (Bilateral Symmetry) qua trục gân chính (Midrib).
- [ ] **2.1.2. Tính tương đồng trong không gian màu sắc (Color Similarity)**
  - Tông màu chủ đạo là dải màu xanh lục do sắc tố diệp lục (Chlorophyll a, b) chi phối.
  - Sự tập trung cao trong không gian màu HSV tại dải Hue xanh (khoảng $60^\circ - 120^\circ$) và kênh Green trong RGB.
- [ ] **2.1.3. Tính chất vùng đóng kín (Closed Contour Topology)**
  - Mỗi lá là một đối tượng liên thông đơn (Single Connected Component) tách biệt hoàn toàn với phông nền (Background).
  - *Hình ảnh minh họa:* Ảnh chụp các lá khác loài nhưng có cùng tông màu xanh và cấu trúc đối xứng.

#### 2.2 Đặc điểm khác nhau giữa các loài lá cây (Kèm so sánh Side-by-Side)
- [ ] **2.2.1. Khác biệt về hình dạng viền bao (Morphological Shape & Contour)**
  - Hình dạng tổng thể phiến lá: Hình bầu dục (Oval), hình mác (Lanceolate), hình khiên/tròn (Orbicular), hình tim (Cordate), hình bàn tay/xẻ thùy (Palmate).
  - So sánh trực quan dạng viền ngoài qua mặt nạ nhị phân (Binary Contour Masks).
- [ ] **2.2.2. Khác biệt về cấu trúc mép lá (Leaf Margin Roughness)**
  - Mép lá trơn (Entire), mép răng cưa nhỏ (Serrulate), răng cưa sắc nhọn (Serrate), mép lượn sóng (Sinuate).
  - *Hình ảnh so sánh Side-by-Side:* Phóng to chi tiết mép viền lá trơn đối sánh với mép răng cưa.
- [ ] **2.2.3. Khác biệt về hệ thống gân lá (Venation Patterns & Surface Texture)**
  - Gân lá hình lông chim (Pinnate), gân lá hình chân vịt/chân vịt lan tỏa (Palmate), gân song song (Parallel).
  - Mức độ nổi gân và độ nhám bề mặt (Texture Roughness).
  - *Hình ảnh so sánh Side-by-Side:* Ảnh xám và phổ biên độ/Gradient làm nổi bật cấu trúc gân lá giữa các loài.
- [ ] **2.2.4. Khác biệt về sắc thái màu sắc và phân bố đốm màu (Color Nuances)**
  - Sự pha trộn sắc tố Carotenoid/Anthocyanin tạo lá đốm, viền tía, lá non pha đỏ đối lập với xanh tuyền đồng nhất.

#### 2.3 Bảng đối sánh hình thái học và đặc trưng thị giác các loài trong CSDL
- [ ] **Bảng đối sánh tổng hợp đa tiêu chí:**
  - Tiêu chí so sánh: Dạng phiến lá, Kiểu mép lá, Hệ gân lá, Tỷ lệ dài/rộng (Aspect Ratio), Màu sắc chủ đạo, Điểm nhận diện đặc trưng nhất.

---

### PHẦN III. XÂY DỰNG CÁC BỘ ĐẶC TRƯNG NHẬN DIỆN LÁ CÂY (YÊU CẦU 2)

#### 3.1 Nhóm Đặc trưng Dựa trên Màu sắc (Color-based Features)
- [ ] **3.1.1. Biểu đồ tần suất màu RGB (RGB Histogram) - Phương pháp cơ sở (Baseline)**
  - *Nguyên lý & Ý nghĩa:* Đếm tần suất xuất hiện mức sáng trên 3 kênh màu Red, Green, Blue; lượng tử hóa thành các bin không gian màu ($8 \times 8 \times 8 = 512$ bins hoặc $32 \times 3 = 96$ bins).
  - *Cơ sở toán học & Công thức chuẩn hóa:*
    $$H_{\text{norm}}(r, g, b) = \frac{H(r, g, b)}{\sum_{r,g,b} H(r, g, b)}$$
  - *Các độ đo tương đồng Histogram kinh điển trong CSDL ĐPT:*
    - **Histogram Intersection (Giao biểu đồ):** $S_{\text{intersect}}(H_1, H_2) = \sum_i \min(H_1(i), H_2(i))$
    - **Chi-Square Distance ($\chi^2$):** $d_{\chi^2}(H_1, H_2) = \sum_i \frac{(H_1(i) - H_2(i))^2}{H_1(i) + H_2(i)}$
    - **Bhattacharyya Distance:** $d_B(H_1, H_2) = \sqrt{1 - \sum_i \sqrt{H_1(i) \cdot H_2(i)}}$
  - *Hạn chế của RGB Histogram:* Nhạy cảm với điều kiện chiếu sáng (nắng/bóng râm), chiều vector lớn, dễ bị chi phối bởi nền nếu không áp dụng mask vùng lá.
- [ ] **3.1.2. Chuyển dịch sang Không gian HSV & Mô-men màu sắc (Color Moments)**
  - *Luận chứng khoa học:* Tách riêng biệt kênh màu sắc thuần túy (Hue) và độ bão hòa (Saturation) khỏi độ chói sáng (Value), đạt tính bất biến với biến thiên cường độ sáng.
  - *Hue Histogram (180 bins):* Phân tích phân bố sắc tố diệp lục thuần túy theo góc quang phổ $0^\circ \rightarrow 180^\circ$.
  - *HSV Color Moments (9D/6D):* Nén thông tin phân bố màu qua các mô-men thống kê:
    - Kỳ vọng trung bình (Mean - $\mu_c$): $\mu_c = \frac{1}{N} \sum_{i=1}^N P_{c,i}$
    - Độ lệch chuẩn (Standard Deviation - $\sigma_c$): $\sigma_c = \sqrt{\frac{1}{N} \sum_{i=1}^N (P_{c,i} - \mu_c)^2}$
    - Độ lệch (Skewness - $s_c$): $s_c = \sqrt[3]{\frac{1}{N} \sum_{i=1}^N (P_{c,i} - \mu_c)^3}$
  - *Giá trị thông tin:* Vector cực kỳ gọn nhẹ (6 đến 9 chiều), tốc độ khớp cực nhanh, kháng nhiễu ánh sáng tối ưu.

#### 3.2 Nhóm Đặc trưng Dựa trên Kết cấu & Gân lá (Texture-based & Venation Features)
- [ ] **3.2.1. Ma trận đồng xuất hiện mức xám GLCM (Gray-Level Co-occurrence Matrix) - Cốt lõi**
  - *Nguyên lý & Ý nghĩa:* Thống kê tần suất xuất hiện đồng thời của các cặp pixel có mức xám $(i, j)$ ở khoảng cách $d$ và góc định hướng $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$.
  - *Các chỉ số thành phần & Công thức toán học:*
    - **Contrast (Độ tương phản gân lá):** $\sum_{i,j} |i - j|^2 P(i,j)$ (Cực kỳ nhạy với gân lá nổi gồ ghề).
    - **Dissimilarity (Độ khác biệt):** $\sum_{i,j} |i - j| P(i,j)$
    - **Homogeneity (Độ đồng nhất):** $\sum_{i,j} \frac{P(i,j)}{1 + (i-j)^2}$ (Phân biệt lá trơn láng với lá gân chằng chịt).
    - **Energy (Năng lượng / ASM):** $\sum_{i,j} P(i,j)^2$
    - **Correlation (Độ tương quan đường gân):** $\sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) P(i,j)}{\sigma_i \sigma_j}$
  - *Giá trị thông tin:* Nắm bắt chính xác hướng phát triển và độ sắc nét của hệ thống gân lá (Venation).
- [ ] **3.2.2. Bộ lọc Gabor đa tần số & đa hướng (Gabor Filter Bank) - Phương pháp nâng cao cho gân lá**
  - *Nguyên lý & Ý nghĩa:* Mô phỏng cơ chế cảm thụ thị giác của vỏ não người bằng sóng điều hòa phẳng nhân với bao hàm Gauss; cực kỳ nhạy bén với các cấu trúc dạng đường rãnh tuyến tính như gân lá.
  - *Cơ sở toán học & Công thức bộ lọc:*
    $$g(x, y; \lambda, \theta, \psi, \sigma, \gamma) = \exp\left(-\frac{x'^2 + \gamma^2 y'^2}{2\sigma^2}\right) \cos\left(2\pi \frac{x'}{\lambda} + \psi\right)$$
    với $x' = x\cos\theta + y\sin\theta$ và $y' = -x\sin\theta + y\cos\theta$.
  - *Tham số cấu hình:* Bước sóng $\lambda$ (độ rộng đường gân), hướng quét $\theta$ ($0^\circ, 45^\circ, 90^\circ, 135^\circ$).
  - *Giá trị thông tin:* Năng lượng đáp ứng trung bình và độ lệch chuẩn của ảnh lọc Gabor tạo vector phân biệt xuất sắc các mạng gân lông chim, gân chân vịt hay gân song song.
- [ ] **3.2.3. Đặc trưng mẫu nhị phân cục bộ LBP (Local Binary Patterns Histogram)**
  - *Nguyên lý & Ý nghĩa:* So sánh mức xám pixel trung tâm với 8 láng giềng xung quanh tạo mã nhị phân 8-bit.
  - *Giá trị thông tin:* Mô tả độ nhám vi mô của biểu bì lá, kháng biến đổi độ sáng đơn điệu.

#### 3.3 Nhóm Đặc trưng Dựa trên Hình dạng & Mép lá (Shape-based & Edge Features)
- [ ] **3.3.1. Bộ chỉ số hình thái học cơ bản (Geometric Morphological Descriptors)**
  - *a) Tỷ lệ khung bao (Aspect Ratio):* $AR = \frac{\text{Length}}{\text{Width}}$ (Phân biệt lá thon dài với lá tròn).
  - *b) Độ tròn (Form Factor / Circularity):* $C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$ (Đạt giá trị 1 với hình tròn tuyệt đối, giảm mạnh khi lá xẻ thùy hoặc mép răng cưa sâu).
  - *c) Độ đặc / Độ lồi (Solidity):* $S = \frac{\text{Area}}{\text{Convex Hull Area}}$ (Phát hiện độ xẻ thùy và khuyết mép lá).
  - *d) Độ lấp đầy khung bao (Extent):* $E = \frac{\text{Area}}{\text{Bounding Box Area}}$.
  - *e) Độ lệch tâm (Eccentricity):* Đo độ kéo dẹt của elip tương đương ôm khít phiến lá.
- [ ] **3.3.2. Bất biến mô-men hình dạng Hu (Hu's 7 Invariant Moments)**
  - *Nguyên lý:* 7 giá trị mô-men hình học phi tuyến tính từ mô-men trung tâm chuẩn hóa $\eta_{p,q}$.
  - *Đặc tính quan trọng:* Bất biến tuyệt đối với 3 phép biến đổi affine: Phép tịnh tiến (Translation), Co giãn (Scale) và Xoay tròn (Rotation).
  - *Giá trị thông tin:* Nhận dạng hình dáng tổng thể của lá bất kể vị trí đặt hay góc xoay trên mặt phẳng quét.
- [ ] **3.3.3. Bộ mô tả cạnh Histogram (Edge Histogram Descriptor - EHD theo chuẩn MPEG-7)**
  - *Nguyên lý:* Chia ảnh thành lưới $4 \times 4 = 16$ ô khối cục bộ (sub-blocks).
  - *Phân loại cạnh:* Tại mỗi khối, sử dụng các bộ lọc phát hiện 5 loại cạnh cơ bản:
    1. Cạnh ngang (Horizontal Edge)
    2. Cạnh dọc (Vertical Edge)
    3. Cạnh chéo $45^\circ$ (Diagonal $45^\circ$ Edge)
    4. Cạnh chéo $135^\circ$ (Diagonal $135^\circ$ Edge)
    5. Cạnh vô hướng (Non-directional Edge)
  - *Giá trị thông tin:* Tạo vector 80 chiều biểu diễn phân bố không gian của các đường viền ngoài và các đường gân lá nổi rõ.
- [ ] **3.3.4. Histogram of Oriented Gradients (HOG - Biểu đồ hướng Gradient)**
  - *Nguyên lý & Ý nghĩa:* Chia ảnh vùng lá thành các ô cell nhỏ (ví dụ $8 \times 8$ pixels), tính độ lớn và hướng của vector đạo hàm không gian, tích lũy vào 9 bins góc hướng ($0^\circ \rightarrow 180^\circ$); chuẩn hóa theo khối Block liên lụy (L2-norm).
  - *Giá trị thông tin:* Bộ mô tả hình dáng viền mép lá mạnh mẽ nhất trong thị giác máy tính truyền thống, phân biệt xuất sắc mép lá trơn, lượn sóng và các kiểu răng cưa nhọn.
- [ ] **3.3.5. Bộ mô tả đặc trưng cục bộ DAISY (DAISY Local Feature Descriptor)**
  - *Nguyên lý & Ý nghĩa:* Lấy cảm hứng từ SIFT nhưng cấu trúc lưới lấy mẫu hình các vòng tròn đồng tâm cánh hoa cúc; sử dụng các bộ lọc đạo hàm làm mờ Gaussian lũy tiến.
  - *Giá trị thông tin:* Tốc độ tính toán nhanh, thích hợp cho việc so khớp dày đặc (dense matching) các điểm uốn khúc viền mép lá và ngã ba gân lá phức tạp.

#### 3.4 Nhóm Đặc trưng Biểu diễn Sâu (Deep Learning Feature Embeddings)
- [ ] **3.4.1. Đặc trưng biểu diễn sâu toàn cục (Pretrained ResNet-18 - 512D Embeddings)**
  - *Nguyên lý:* Sử dụng mạng tích chập sâu ResNet-18 huấn luyện trước trên tập ImageNet, trích xuất vector 512 chiều tại tầng Global Average Pooling.
  - *Giá trị thông tin:* Nắm bắt các đặc trưng trừu tượng mức cao (High-level Semantic Features) kết hợp hài hòa cả màu sắc, hình dáng và kết cấu.
- [ ] **3.4.2. Đặc trưng biểu diễn sâu có giám sát (Fine-tuned ResNet-18 - 512D Embeddings)**
  - *Nguyên lý:* Huấn luyện tinh chỉnh (Transfer Learning & Fine-tuning) trên chính tập dữ liệu lá cây có nhãn loài.
  - *Giá trị thông tin:* Ép các mẫu cùng loài co cụm chặt chẽ trong không gian Hilbert và đẩy xa các loài khác biệt, tối ưu hóa độ phân tách cho các cặp loài có ngoại hình tương tự nhau.

#### 3.5 Thiết kế các Vector Đặc trưng Tổng hợp & Chiến lược Chuẩn hóa
- [ ] **3.5.1. BỘ 1 – Vector Đặc trưng Truyền thống Cốt lõi (Handcrafted Feature Vector - 25 chiều)**
  - *Cấu trúc:* 6D HSV Color Moments + 11D Shape (Hu 7D + 4 Chỉ số hình thái) + 8D Texture (GLCM 6D + Gray Stats 2D).
  - *Đặc điểm:* Gọn nhẹ, tính toán thời gian thực, có tính giải thích hình thái rõ ràng (Explainable AI).
- [ ] **3.5.2. BỘ 2 – Vector Đặc trưng Mở rộng Truyền thống (Extended Handcrafted Vector)**
  - *Cấu trúc:* Bổ sung HOG (hoặc Edge Histogram / Gabor Filter) kết hợp với Bộ 1.
  - *Đặc điểm:* Gia tăng đáng kể năng lực phân biệt các chi tiết vi mô trên mép lá và gân lá.
- [ ] **3.5.3. BỘ 3 – Vector Đặc trưng Học sâu Toàn cục (Deep Learning Embeddings - 512 chiều)**
  - *Cấu trúc:* 512 chiều ResNet-18 (Pretrained hoặc Fine-tuned).
- [ ] **3.5.4. BỘ 4 – Vector Đặc trưng Đa tầng Kết hợp (Hybrid/Fused Feature Vector - 537 chiều)**
  - *Cấu trúc:* Ghép nối (Concatenation) Bộ 1 (25D) và Bộ 3 (512D).
  - *Đặc điểm:* Kết hợp đồng thời độ nhạy hình học biên của Handcrafted và độ trừu tượng ngữ nghĩa của Deep Learning.
- [ ] **3.5.5. Chiến lược Chuẩn hóa & Giảm chiều Dữ liệu**
  - *Chuẩn hóa Z-Score & L2-norm:* Đồng bộ thang đo giữa các đặc trưng tỷ lệ, diện tích và mức xám.
  - *Chiếu giảm chiều PCA (Principal Component Analysis 2D & 3D):* Giữ lại $\ge 90\%-95\%$ tổng phương sai để trực quan hóa không gian dữ liệu.
  - *Phân tích gom cụm K-Means Clustering:* Đánh giá cấu trúc cụm tự nhiên của các loài lá cây.

---

### PHẦN KẾT LUẬN & HƯỚNG PHÁT TRIỂN
- [ ] **Tóm tắt kết quả đạt được:**
  - Hoàn thành tập dữ liệu $\ge 1.000$ ảnh chuẩn hóa đáp ứng đúng quy chuẩn Yêu cầu 1.
  - Xây dựng hệ thống các bộ đặc trưng toàn diện (Color-based, Texture-based, Shape-based, Deep Embeddings) đáp ứng đầy đủ Yêu cầu 2.
- [ ] **Kế hoạch triển khai cho giai đoạn tiếp theo (Yêu cầu 3 & 4):**
  - Thiết kế cấu trúc lưu trữ và đánh chỉ mục vector trong CSDL Đa phương tiện.
  - Xây dựng thuật toán tính khoảng cách (Euclidean, Cosine, Chi-Square, Manhattan) phục vụ truy vấn Top-K tương đồng (Query by Example) trên giao diện tương tác.

---

## 📊 DANH SÁCH BẢNG BIỂU VÀ HÌNH ẢNH MINH HỌA CẦN CHUẨN BỊ CHO BÁO CÁO

### 1. Danh sách Bảng biểu (Tables):
1. **Bảng 1:** Danh mục mã định danh (Leaf_ID) và quy chuẩn tên file cho các loài lá cây.
2. **Bảng 2:** Lược đồ Metadata quản lý Đối tượng Ảnh lá cây trong CSDL (Fields, Data Type, Description).
3. **Bảng 3:** Bảng tổng hợp số lượng mẫu và thông số kích thước, tỷ lệ của các loài trong dataset.
4. **Bảng 4:** Bảng đối sánh hình thái học và đặc trưng thị giác các loài lá cây (Side-by-side comparison).
5. **Bảng 5:** Bảng phân rã chi tiết cấu trúc các vector đặc trưng (Color-based, Texture-based, Shape-based, Deep Learning).
6. **Bảng 6:** Bảng so sánh ưu/nhược điểm giữa các phương pháp trích xuất (RGB Histogram vs HSV Moments, GLCM vs Gabor, Hu Moments vs HOG/DAISY).

### 2. Danh sách Hình ảnh & Biểu đồ (Figures):
1. **Hình 1:** Sơ đồ cấu trúc cây thư mục lưu trữ vật lý của bộ dữ liệu ảnh lá cây.
2. **Hình 2:** Giao diện bảng tính quản lý Metadata của Đối tượng Ảnh lá cây.
3. **Hình 3:** Biểu đồ phân bố số lượng ảnh giữa các loài trong bộ dữ liệu (đánh giá cân bằng mẫu).
4. **Hình 4:** Biểu đồ phân bố kích thước phiến lá và tỷ lệ dài/rộng (Aspect Ratio).
5. **Hình 5:** Minh họa trực quan các đặc điểm giống nhau về cấu tạo giải phẫu và màu sắc diệp lục.
6. **Hình 6:** So sánh trực quan Side-by-Side: Viền bao ngoài (Contour) và mép lá răng cưa vs mép lá trơn.
7. **Hình 7:** So sánh trực quan Side-by-Side: Cấu trúc gân lá (Venation) qua xử lý ảnh xám & gradient.
8. **Hình 8:** Minh họa trực quan các bộ lọc đặc trưng: Phản ứng của Gabor Filter Bank trên gân lá và Biểu đồ định hướng HOG trên mép lá.
9. **Hình 9:** So sánh biểu đồ màu sắc: RGB Histogram đối sánh với Hue Histogram trong không gian HSV.
10. **Hình 10:** Biểu đồ chiếu giảm chiều PCA 2D/3D phân bố các loài lá cây trong không gian đặc trưng.
11. **Hình 11:** Kết quả gom cụm K-Means thể hiện độ tách biệt giữa các nhóm loài lá cây.
