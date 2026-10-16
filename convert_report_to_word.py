"""
convert_report_to_word.py
Tạo file Word (.docx) chuyên nghiệp từ báo cáo chi tiết, tích hợp bảng biểu và nhúng toàn bộ ảnh minh họa thực tế.
"""
import os
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCX_PATH = os.path.join(PROJECT_ROOT, "BAO_CAO_CHI_TIET_BTL_HCSDLDPT.docx")

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def create_word_report():
    doc = Document()
    
    # Page setup - Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(30, 41, 59) # Slate 800

    # TITLE
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("BÁO CÁO CHI TIẾT BÀI TẬP LỚN\nHỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN (MMDBMS)")
    title_run.bold = True
    title_run.font.size = Pt(18)
    title_run.font.color.rgb = RGBColor(15, 23, 42) # Slate 900
    title_p.paragraph_format.space_after = Pt(6)

    # SUBTITLE
    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub_p.add_run("Đề tài: Hệ thống Tìm kiếm Ảnh Lá Cây Tương đồng dựa trên Nội dung (Leaf CBIR)\nGiải quyết chuyên sâu: Yêu cầu 1 (Bộ dữ liệu) & Yêu cầu 2 (Bộ đặc trưng Handcrafted & Deep Features)")
    sub_run.italic = True
    sub_run.font.size = Pt(12)
    sub_run.font.color.rgb = RGBColor(13, 148, 136) # Teal 600
    sub_p.paragraph_format.space_after = Pt(20)

    # Line separator
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(14)
    run_div = p_div.add_run("―" * 55)
    run_div.font.color.rgb = RGBColor(203, 213, 225)
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # HELPER HEADERS
    def add_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(8)
        r = h.add_run(text)
        r.bold = True
        r.font.size = Pt(15)
        r.font.color.rgb = RGBColor(15, 118, 110) # Teal 700
        return h

    def add_h2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        r = h.add_run(text)
        r.bold = True
        r.font.size = Pt(13)
        r.font.color.rgb = RGBColor(30, 58, 138) # Blue 900
        return h

    def add_h3(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(4)
        r = h.add_run(text)
        r.bold = True
        r.font.size = Pt(12)
        r.font.color.rgb = RGBColor(51, 65, 85) # Slate 700
        return h

    def add_p(text, bold_prefix="", italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.bold = True
            rb.font.color.rgb = RGBColor(15, 23, 42)
        r = p.add_run(text)
        if italic:
            r.italic = True
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.bold = True
        p.add_run(text)
        return p

    # ══════════════════════════════════════════════════════════════════
    # PHẦN 1
    # ══════════════════════════════════════════════════════════════════
    add_h1("PHẦN 1: THU THẬP VÀ MÔ TẢ ĐẶC TÍNH BỘ DỮ LIỆU ẢNH LÁ CÂY")
    
    add_h2("1.1. Tiêu chuẩn và Quy cách Thu thập Bộ Dữ liệu")
    add_p("Trong các hệ thống Cơ sở dữ liệu đa phương tiện (MMDBMS), chất lượng và sự nhất quán của tập ảnh đầu vào đóng vai trò quyết định đến độ chính xác của các thuật toán trích xuất đặc trưng và đo khoảng cách tương đồng. Bộ dữ liệu ảnh lá cây được xây dựng nghiêm ngặt theo các tiêu chuẩn kỹ thuật sau:")
    add_bullet(" Đạt tổng cộng 1240 ảnh (gồm 1200 ảnh trong CSDL tra cứu và 40 ảnh truy vấn kiểm thử độc lập), vượt yêu cầu tối thiểu 1000 ảnh.", "1. Số lượng ảnh: ")
    add_bullet(" Tất cả các mẫu vật đều được đặt phẳng trên nền chụp, máy ảnh đặt vuông góc 90° nhìn thẳng từ trên xuống. Góc chụp thẳng đứng này loại bỏ hoàn toàn méo góc phối cảnh (perspective distortion).", "2. Góc chụp thẳng đứng (Top-down Vertical View): ")
    add_bullet(" Mỗi bức ảnh chỉ chứa duy nhất một phiến lá cây đại diện, đặt cân đối ở trung tâm khung hình, xung quanh là nền tương phản cao.", "3. Phân tách đơn vật thể: ")
    add_bullet(" Toàn bộ 1240 ảnh được định dạng chuẩn JPEG (24-bit RGB), đồng nhất kích thước chuẩn 416 x 416 pixels (tỉ lệ khung hình vuông 1:1). Lá cây chiếm diện tích cân xứng từ 40% - 75% diện tích ảnh.", "4. Quy cách lưu trữ & tỉ lệ khung hình: ")

    add_h2("1.2. Thống kê Định lượng Bộ Dữ liệu (Quantitative Statistics)")
    add_p("Bộ dữ liệu bao gồm 8 loài cây nông nghiệp và cây ăn quả phổ biến, đại diện cho nhiều cấu trúc hình thái sinh học khác nhau. Dữ liệu có tính cân bằng tuyệt đối (150 ảnh/loài trong CSDL):")

    # TABLE 1: Class distribution
    table1 = doc.add_table(rows=1, cols=6)
    table1.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table1.rows[0].cells
    hdr_titles = ["STT", "Loài cây (English)", "Tên Tiếng Việt", "Tên Khoa Học", "Số lượng CSDL", "Tỉ lệ"]
    for i, title in enumerate(hdr_titles):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "0F766E") # Teal header
        for run in hdr_cells[i].paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.bold = True

    classes_info = [
        ("1", "Apple", "Cây Táo tây", "Malus domestica", "150", "12.5%"),
        ("2", "Cherry", "Cây Anh đào", "Prunus avium", "150", "12.5%"),
        ("3", "Grape", "Cây Nho", "Vitis vinifera", "150", "12.5%"),
        ("4", "Peach", "Cây Đào", "Prunus persica", "150", "12.5%"),
        ("5", "Pepper", "Cây Ớt chuông", "Capsicum annuum", "150", "12.5%"),
        ("6", "Potato", "Cây Khoai tây", "Solanum tuberosum", "150", "12.5%"),
        ("7", "Strawberry", "Cây Dâu tây", "Fragaria × ananassa", "150", "12.5%"),
        ("8", "Tomato", "Cây Cà chua", "Solanum lycopersicum", "150", "12.5%")
    ]
    for row_data in classes_info:
        row_cells = table1.add_row().cells
        for col_idx, val in enumerate(row_data):
            row_cells[col_idx].text = val

    add_p("")
    add_h3("Bảng Thống kê Chỉ số Sinh học Thực tế (Trích xuất từ 1200 ảnh CSDL):")
    
    # TABLE 2: Features stats
    table2 = doc.add_table(rows=1, cols=6)
    table2.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table2.rows[0].cells
    hdr_titles2 = ["Loài lá", "Sắc độ Hue (0-180)", "Bão hòa Sat (0-255)", "Độ tròn Circularity", "Độ lồi Solidity", "GLCM Contrast"]
    for i, title in enumerate(hdr_titles2):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1E3A8A") # Navy header
        for run in hdr_cells[i].paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.bold = True

    stats_info = [
        ("Apple", "59.1 ± 6.8", "90.7 ± 18.2", "0.58 ± 0.08", "0.90 ± 0.03", "4.03 ± 1.45"),
        ("Cherry", "48.0 ± 5.9", "112.0 ± 20.4", "0.56 ± 0.07", "0.90 ± 0.02", "3.06 ± 1.12"),
        ("Grape", "45.3 ± 7.1", "132.5 ± 22.1", "0.45 ± 0.06", "0.85 ± 0.04", "5.68 ± 1.80"),
        ("Peach", "39.2 ± 8.4", "141.5 ± 25.3", "0.44 ± 0.05", "0.87 ± 0.03", "2.89 ± 0.95"),
        ("Pepper", "50.8 ± 6.2", "114.7 ± 19.5", "0.66 ± 0.06", "0.92 ± 0.02", "5.62 ± 1.63"),
        ("Potato", "41.5 ± 7.6", "101.9 ± 23.0", "0.66 ± 0.07", "0.95 ± 0.02", "6.22 ± 1.94"),
        ("Strawberry", "43.4 ± 6.5", "126.0 ± 21.8", "0.61 ± 0.08", "0.93 ± 0.03", "6.72 ± 2.10"),
        ("Tomato", "46.9 ± 7.3", "116.6 ± 20.9", "0.47 ± 0.07", "0.85 ± 0.05", "4.81 ± 1.72")
    ]
    for row_data in stats_info:
        row_cells = table2.add_row().cells
        for col_idx, val in enumerate(row_data):
            row_cells[col_idx].text = val

    add_p("")
    # Add Image 11: Dataset overview
    img11_path = os.path.join(PROJECT_ROOT, "visualization", "11_dataset_leaf_species_grid.png")
    if os.path.exists(img11_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        doc.add_picture(img11_path, width=Inches(6.2))
        add_p("Hình 1: Tổng quan 8 loài lá cây trong tập dữ liệu (ảnh chuẩn hóa 416 x 416 pixels).", italic=True)

    add_h2("1.3. Phân tích Dưới góc nhìn Cảm nhận Thị giác (Visual & Perceptual Analysis)")
    add_p("Khi quan sát bằng mắt thường, các loài lá cây bộc lộ các quy luật thị giác vừa thống nhất vừa phân hóa sâu sắc:")
    add_bullet(" Bố cục đơn lập ở trung tâm, cấu trúc đối xứng hai bên trục gân chính (Bilateral Symmetry) và màu xanh lục chi phối do sắc tố quang hợp Chlorophyll.", "A. Điểm tương đồng chung: ")
    add_bullet(" Phiến lá biến thiên từ dạng thuôn mác rất hẹp (Peach) đến hình bầu dục tròn trịa (Pepper, Potato), và dạng xẻ thùy phức tạp (Grape, Tomato). Mép lá chia thành mép nguyên (Pepper, Peach), mép răng cưa to sắc cạnh (Strawberry) và mép răng cưa mịn (Apple, Cherry). Gân lá phân hóa giữa gân lông chim (Pinnate) và gân chân vịt xòe bàn tay (Palmate ở lá Nho).", "B. Điểm khác biệt sâu sắc: ")

    add_h2("1.4. Đối chiếu So sánh Trực quan Side-by-Side giữa các Cặp Loài Đặc trưng")
    add_p("Để làm rõ khả năng phân biệt của thị giác, chúng tôi thiết lập 4 cặp so sánh đối đầu kinh điển:")
    add_bullet(" Cả hai đều có phiến lá đơn, mép nguyên trơn láng. Khác nhau: Lá Đào thon nhọn như mũi mác (Extent chỉ 0.45, Circularity 0.44), trong khi lá Ớt chuông bầu bĩnh hình trứng rộng (Extent đạt 0.60, Circularity 0.66).", "Cặp 1: Peach (Đào) vs Pepper (Ớt chuông) - ")
    add_bullet(" Cả hai đều có đường viền xẻ thùy sâu và độ lồi Solidity thấp (~0.85). Khác nhau: Lá Nho có hệ gân chân vịt xòe bàn tay từ gốc cuống hình tim; lá Cà chua có hệ gân lông chim với các thùy lá xẻ bất đối xứng.", "Cặp 2: Grape (Nho) vs Tomato (Cà chua) - ")
    add_bullet(" Cùng có dạng phiến lá rộng, tỉ lệ khung hình gần vuông (~0.95). Khác nhau: Lá Dâu tây có răng cưa tam giác rất to và gân phụ nổi cộm sần sùi (GLCM Contrast 6.72); lá Táo có mép răng cưa mịn màng, bề mặt phẳng hơn nhiều (Contrast 4.03).", "Cặp 3: Strawberry (Dâu tây) vs Apple (Táo) - ")
    add_bullet(" Khung hình oval elip tương đồng. Khác nhau: Bề mặt lá Anh đào phủ lớp sáp bóng nhẵn mịn màng (Homogeneity 0.63); lá Khoai tây có bề mặt thô nhám, tán xạ mờ (Homogeneity 0.52).", "Cặp 4: Cherry (Anh đào) vs Potato (Khoai tây) - ")

    # Add Image 12: Side-by-side
    img12_path = os.path.join(PROJECT_ROOT, "visualization", "12_side_by_side_visual_comparison.png")
    if os.path.exists(img12_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        doc.add_picture(img12_path, width=Inches(6.2))
        add_p("Hình 2: Đối chiếu trực quan Side-by-Side 4 cặp loài lá cây đặc trưng.", italic=True)

    # ══════════════════════════════════════════════════════════════════
    # PHẦN 2
    # ══════════════════════════════════════════════════════════════════
    add_h1("PHẦN 2: XÂY DỰNG CÁC BỘ ĐẶC TRƯNG NHẬN DIỆN LÁ CÂY (FEATURE ENGINEERING)")

    add_h2("2.1. Kiến trúc Tiền xử lý & Phân đoạn Viền Sắc nét bằng AI (BiRefNet_lite)")
    add_p("Để trích xuất đặc trưng hình thái học chính xác, hệ thống tích hợp mô hình AI phân đoạn SOTA BiRefNet_lite (Bilateral Reference Network). Mô hình tiếp nhận ảnh đầu vào, bóc tách chính xác toàn bộ cuống lá và viền răng cưa, loại bỏ triệt để bóng đổ của lá trên mặt nền. Từ mặt nạ nhị phân đầu ra M(x, y), thuật toán Suzuki & Abe trích xuất đường viền khép kín lớn nhất phục vụ tính toán hình học.")

    add_h2("2.2. Không gian Đặc trưng Truyền thống 25 Chiều (25D Handcrafted Feature Vector)")
    add_p("Hệ thống xây dựng bộ vector 25 chiều gồm 3 nhóm đặc trưng toán học độc lập:")

    add_h3("2.2.1. Nhóm Đặc trưng Màu sắc (Color Descriptors - 6 Chiều)")
    add_p("Ảnh được chuyển đổi từ không gian RGB sang không gian màu HSV (Hue - Saturation - Value) nhằm tách biệt độ sáng (Value) với sắc thái màu (Hue) và độ tinh khiết (Saturation). Trên vùng mặt nạ lá cây, tính toán mô-men màu bậc 1 (Mean) và bậc 2 (Standard Deviation) trên cả 3 kênh:")
    add_bullet(" μ_c = (1 / N_leaf) * ∑ I_c(x, y)", "• Giá trị trung bình: ")
    add_bullet(" σ_c = sqrt( (1 / N_leaf) * ∑ (I_c(x, y) - μ_c)^2 )", "• Độ lệch chuẩn: ")
    add_p("Vector màu sắc 6D: f_color = [μ_H, σ_H, μ_S, σ_S, μ_V, σ_V]. Giá trị thông tin: Phân biệt sắc thái diệp lục (xanh vàng của lá Đào vs xanh thẫm của lá Táo) và độ đậm đà của sắc tố sinh học.")

    add_h3("2.2.2. Nhóm Đặc trưng Hình dạng Hình học (Shape Descriptors - 11 Chiều)")
    add_p("Hình dạng là đặc trưng mang tính di truyền cao nhất của thực vật, bao gồm:")
    add_bullet(" 7 phương trình mô-men phi tuyến bất biến hoàn toàn với phép tịnh tiến, co giãn tỉ lệ và phép quay trong mặt phẳng 2D. Áp dụng biến đổi logarit cơ số 10 có dấu: h_k = -sign(φ_k) * log10(|φ_k| + 1e-12) để ổn định dải đo.", "A. 7 Bất biến Hu Moments (h_1 đến h_7): ")
    add_bullet(" AR = W_box / H_box (phân biệt lá dài hẹp với lá tròn bè).", "B. Tỉ lệ khung hình (Aspect Ratio): ")
    add_bullet(" C = (4 * π * Area) / (Perimeter^2). Giá trị trong khoảng (0, 1]. Hình tròn hoàn hảo có C = 1. Lá càng xẻ thùy hoặc mép răng cưa sâu thì chu vi Perimeter càng lớn khiến C giảm mạnh.", "C. Độ tròn (Circularity / Compactness): ")
    add_bullet(" S = Area / Area_ConvexHull. Nhận diện mức độ xẻ thùy lõm. Lá nguyên đạt S > 0.92, lá xẻ thùy (Nho, Cà chua) tụt xuống S ≈ 0.85.", "D. Độ lồi (Solidity): ")
    add_bullet(" E = Area / (W_box * H_box) (đo mức độ chiếm khoảng không của phiến lá).", "E. Tỉ lệ chiếm diện tích hộp (Extent): ")
    add_p("Vector hình dạng 11D: f_shape = [h_1, h_2, h_3, h_4, h_5, h_6, h_7, AR, C, S, E].")

    add_h3("2.2.3. Nhóm Đặc trưng Kết cấu Gân lá (GLCM Texture Descriptors - 8 Chiều)")
    add_p("Sử dụng Ma trận Đồng mức xám GLCM (Gray-Level Co-occurrence Matrix) của Haralick. Ảnh mức xám được lượng tử hóa về 32 mức, tính toán xác suất đồng thời trên 4 hướng (0°, 45°, 90°, 135°) và 2 khoảng cách (d=1, d=3) để triệt tiêu phụ thuộc hướng xoay. 6 chỉ số Haralick và 2 chỉ số mức xám gồm:")
    add_bullet(" Đo độ tương phản mức xám giữa các pixel lân cận. Lá gân nổi gồ ghề (Dâu tây: 6.72) có Contrast cao; lá nhẵn bóng (Đào: 2.89) có Contrast rất thấp.", "1. GLCM Contrast: ")
    add_bullet(" Đo độ dị biệt tuyến tính.", "2. GLCM Dissimilarity: ")
    add_bullet(" Đo tính đồng nhất cục bộ (Cherry nhẵn bóng đạt 0.63, Potato nhám đạt 0.52).", "3. GLCM Homogeneity: ")
    add_bullet(" Đo tính trật tự và lặp lại của hoa văn kết cấu.", "4. GLCM Energy & ASM: ")
    add_bullet(" Thể hiện sự liên kết tuyến tính theo dải gân lá.", "5. GLCM Correlation: ")
    add_bullet(" Đo cường độ hấp thụ sáng và biến thiên mức xám biểu bì.", "6. Gray Mean & Gray Std: ")
    add_p("Vector kết cấu 8D: f_texture = [Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM, Gray_Mean, Gray_Std].")

    add_h2("2.3. Chuẩn hóa Z-Score & Hàm Đo Khoảng Cách Trọng Số")
    add_p("Do các thuộc tính có đơn vị đo hoàn toàn khác nhau, toàn bộ không gian 25 chiều được chuẩn hóa Z-Score: z = (f - μ) / σ (đưa về phân phối chuẩn μ=0, σ=1).")
    add_p("Khi tìm kiếm, khoảng cách Euclidean có trọng số được tính toán linh hoạt:")
    add_p("D_total = w_shape * D_shape + w_texture * D_texture + w_color * D_color", bold_prefix="Hàm khoảng cách tổng hợp: ")
    add_p("Khoảng cách được quy đổi sang độ tương đồng phần trăm [0%, 100%] bằng hàm phân rã mũ: Similarity(%) = 100 * exp(-D_total / 1.5).")

    add_h2("2.4. Đối chứng Nâng cao: Handcrafted Features vs Deep Learning Features (ResNet-18)")
    add_p("Hệ thống đã triển khai thực nghiệm đối chứng toàn diện 3 cấp độ:")
    add_bullet(" Trích xuất 25D thủ công, cho phép người dùng tùy biến trọng số tìm kiếm theo ý muốn (Explainability rất cao). Precision@5 đạt 50.0%.", "• Cấp 1 (Handcrafted 25D): ")
    add_bullet(" Dùng mạng ResNet-18 ImageNet tiền huấn luyện làm bộ trích xuất vector 512D không cần nhãn (Zero-shot Transfer Learning). Precision@5 tăng vọt lên 82.0%.", "• Cấp 2 (Pretrained ResNet-18): ")
    add_bullet(" Huấn luyện tinh chỉnh ResNet-18 trực tiếp trên tập 1200 ảnh lá cây trong 15 epoch trên GPU RTX 4060 (Validation Acc đạt 99.58%). Precision@5 và mAP đạt 100.0% tuyệt đối, K-Means Cluster Purity đạt 99.92%!", "• Cấp 3 (Fine-tuned ResNet-18): ")

    # TABLE 3: Benchmark comparison
    table3 = doc.add_table(rows=1, cols=4)
    table3.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells3 = table3.rows[0].cells
    hdr_titles3 = ["Chỉ số đánh giá", "Cấp 1: Handcrafted 25D", "Cấp 2: Pretrained 512D", "Cấp 3: Fine-tuned 512D"]
    for i, title in enumerate(hdr_titles3):
        hdr_cells3[i].text = title
        set_cell_background(hdr_cells3[i], "047857") # Emerald header
        for run in hdr_cells3[i].paragraphs[0].runs:
            run.font.color.rgb = RGBColor(255, 255, 255)
            run.bold = True

    bench_data = [
        ("Precision@1 (Top-1)", "52.5%", "87.5%", "100.0% (Tuyệt đối)"),
        ("Precision@3 (Top-3)", "53.3%", "83.3%", "100.0% (Tuyệt đối)"),
        ("Precision@5 (Top-5)", "50.0%", "82.0%", "100.0% (Tuyệt đối)"),
        ("mAP@5 (Mean Average Precision)", "62.9%", "91.0%", "100.0% (Tuyệt đối)"),
        ("Cluster Purity (K-Means k=8)", "36.8%", "59.2%", "99.92%"),
        ("Adjusted Rand Index (ARI)", "0.1272", "0.3613", "0.9981 (Khớp nhãn hoàn hảo)"),
        ("Silhouette Score", "0.1043", "0.0726", "0.4853 (Cụm cô đặc vượt trội)"),
        ("Tính giải thích (Explainability)", "Rất cao (Hiểu rõ độ tròn, gân, màu)", "Thấp (Vector ẩn)", "Thấp (Vector ẩn)")
    ]
    for row_data in bench_data:
        row_cells = table3.add_row().cells
        for col_idx, val in enumerate(row_data):
            row_cells[col_idx].text = val

    add_p("")
    # Add Image 08: Benchmark 3 levels
    img8_path = os.path.join(PROJECT_ROOT, "visualization", "08_benchmark_3_levels_comparison.png")
    if os.path.exists(img8_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        doc.add_picture(img8_path, width=Inches(6.2))
        add_p("Hình 3: Biểu đồ so sánh đối chứng toàn diện hiệu năng 3 cấp độ phương pháp.", italic=True)

    # Add Image 09: PCA 3D view
    img9_path = os.path.join(PROJECT_ROOT, "visualization", "09_pca_3d_view.png")
    if os.path.exists(img9_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        doc.add_picture(img9_path, width=Inches(6.2))
        add_p("Hình 4: Không gian vector đặc trưng PCA 3D: Pretrained (tản mác) vs Fine-tuned (gom 8 cụm độc lập).", italic=True)

    # ══════════════════════════════════════════════════════════════════
    # KẾT LUẬN
    # ══════════════════════════════════════════════════════════════════
    add_h1("KẾT LUẬN VÀ GIÁ TRỊ THỰC TIỄN")
    add_p("Báo cáo đã giải quyết trọn vẹn và chuyên sâu 2 yêu cầu nền tảng của môn học Cơ Sở Dữ Liệu Đa Phương Tiện:")
    add_bullet(" Đã thu thập và chuẩn hóa thành công 1240 ảnh lá cây chất lượng cao (416x416, 90° top-down, cân bằng 8 loài). Phân tích toàn diện cả về góc độ cảm nhận thị giác con người và các thông số đo lường sinh học thực nghiệm.", "1. Về Bộ Dữ Liệu: ")
    add_bullet(" Đã xây dựng thành công bộ đặc trưng 25D Handcrafted vững chắc về mặt toán học (HSV, Hu Moments, Circularity, Solidity, GLCM), cho phép người dùng giải thích tường minh và linh hoạt tùy biến trọng số tìm kiếm trên giao diện Web. Mở rộng đối chứng xuất sắc với mạng nơ-ron sâu ResNet-18 đưa hiệu năng tìm kiếm lên 100.0%.", "2. Về Bộ Đặc Trưng: ")

    doc.save(DOCX_PATH)
    print(f"Đã tạo thành công file Word tại: {DOCX_PATH}")

if __name__ == '__main__':
    create_word_report()
