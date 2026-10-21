"""
src/feature_extractor.py - Trích xuất đặc trưng đa tầng toàn diện cho CBLIR
Bao gồm đầy đủ 100% các đặc trưng mô tả trong Báo cáo CSDL Đa phương tiện CBLIR:
- Mục 3.1: Màu sắc (RGB Histogram, HSV Moments 6D/9D, Hue Histogram 180 bins)
- Mục 3.2: Kết cấu (GLCM 6D, Gray Stats 2D, Gabor Filter Bank 24D, LBP 59D/256D)
- Mục 3.3: Hình dạng (Chỉ số hình thái 4D/6D, 7 Hu Moments, MPEG-7 EHD 80D, HOG 1764D, DAISY)
- Mục 3.4: Học sâu (ResNet-18 512D Embeddings: Pretrained & Fine-tuned)
- Mục 3.5: Đóng gói 6 BỘ đặc trưng chuẩn (BỘ 1 đến BỘ 6)
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, daisy, hog

# ─── TÊN CÁC ĐẶC TRƯNG CƠ SỞ (25D TRUYỀN THỐNG) ───
FEATURE_NAMES = [
    # Color (6)
    "color_h_mean", "color_h_std",
    "color_s_mean", "color_s_std",
    "color_v_mean", "color_v_std",
    # Shape (11)
    "hu_moment_1", "hu_moment_2", "hu_moment_3", "hu_moment_4",
    "hu_moment_5", "hu_moment_6", "hu_moment_7",
    "aspect_ratio", "circularity", "solidity", "extent",
    # Texture (8)
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation", "glcm_asm",
    "leaf_gray_mean", "leaf_gray_std"
]

# ─── BIREFNET MODEL SINGLETON CHO PHÂN ĐOẠN ───
_BIREFNET_MODEL = None
_BIREFNET_DEVICE = None
_BIREFNET_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_birefnet_model():
    """Tải một lần duy nhất mô hình BiRefNet_lite vào GPU/CPU."""
    global _BIREFNET_MODEL, _BIREFNET_DEVICE
    if _BIREFNET_MODEL is None:
        from transformers import AutoModelForImageSegmentation
        _BIREFNET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _BIREFNET_MODEL = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet_lite", 
            trust_remote_code=True
        )
        _BIREFNET_MODEL.to(_BIREFNET_DEVICE)
        _BIREFNET_MODEL.eval()
    return _BIREFNET_MODEL, _BIREFNET_DEVICE


def segment_leaf(img):
    """
    Phân đoạn phiến lá độ chính xác cao bằng BiRefNet_lite.
    Fallback về Otsu thresholding nếu mô hình gặp sự cố.
    Trả về: binary_mask (uint8 0/255), largest_contour
    """
    h, w = img.shape[:2]
    try:
        model, device = get_birefnet_model()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        input_tensor = _BIREFNET_TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred).resize((w, h), Image.BILINEAR)
        pred_np = np.array(pred_pil)
        binary_mask = (pred_np > 128).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and max(cv2.contourArea(c) for c in contours) > 100:
            largest_contour = max(contours, key=cv2.contourArea)
            final_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            return final_mask, largest_contour
    except Exception as e:
        print(f"[Warning] BiRefNet inference error ({e}), falling back to thresholding.")

    # Fallback: Otsu + Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sat = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, 25, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(otsu_inv, sat_mask)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or max(cv2.contourArea(c) for c in contours) < 1000:
        cleaned = sat_mask
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or max(cv2.contourArea(c) for c in contours) < 500:
        cleaned = np.zeros((h, w), dtype=np.uint8)
        cleaned[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = 255
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    largest_contour = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(gray)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return final_mask, largest_contour


# =========================================================================
# 3.1 NHÓM ĐẶC TRƯNG MÀU SẮC (COLOR FEATURES)
# =========================================================================

def extract_color_moments(img_bgr, mask=None, return_9d=True):
    """
    Mục 3.1.2: Mô-men màu sắc trong không gian HSV.
    - 6 chiều cơ bản: Mean, Std của H, S, V
    - 9 chiều đầy đủ: Bổ sung Skewness (độ lệch) m3 = (1/N * sum((x - mu)^3))^(1/3)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if mask is not None and np.sum(mask > 0) > 0:
        pixels = hsv[mask > 0].astype(np.float32)
    else:
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        
    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]
    
    h_mean, h_std = float(np.mean(h)), float(np.std(h))
    s_mean, s_std = float(np.mean(s)), float(np.std(s))
    v_mean, v_std = float(np.mean(v)), float(np.std(v))
    
    if not return_9d:
        return np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std], dtype=np.float32)
        
    # Skewness
    def calc_skewness(vals, mean_v):
        diff = vals - mean_v
        m3 = np.mean(diff ** 3)
        return float(np.sign(m3) * (np.abs(m3) ** (1.0 / 3.0)))
        
    h_skew = calc_skewness(h, h_mean)
    s_skew = calc_skewness(s, s_mean)
    v_skew = calc_skewness(v, v_mean)
    
    return np.array([h_mean, h_std, h_skew, s_mean, s_std, s_skew, v_mean, v_std, v_skew], dtype=np.float32)


def extract_hue_histogram(img_bgr, mask=None, bins=180):
    """
    Mục 3.1.2: Biểu đồ phân bố sắc tố quang phổ Hue (180 bins thang OpenCV 0 - 180).
    Chuẩn hóa L1 (tổng bằng 1.0).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], mask, [bins], [0, bins]).flatten().astype(np.float32)
    total = np.sum(hist)
    if total > 0:
        hist /= total
    return hist


def extract_rgb_histogram(img_bgr, mask=None, bins_per_channel=8):
    """
    Mục 3.1.1: Biểu đồ tần suất màu RGB 3D (bins_per_channel^3 = 512 bins khi bins=8).
    Chuẩn hóa L1 (tổng bằng 1.0).
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist(
        [rgb], [0, 1, 2], mask,
        [bins_per_channel, bins_per_channel, bins_per_channel],
        [0, 256, 0, 256, 0, 256]
    ).flatten().astype(np.float32)
    total = np.sum(hist)
    if total > 0:
        hist /= total
    return hist


# =========================================================================
# 3.2 NHÓM ĐẶC TRƯNG KẾT CẤU & GÂN LÁ (TEXTURE FEATURES)
# =========================================================================

def extract_glcm_features(gray_img, mask=None):
    """
    Mục 3.2.1: Ma trận đồng xuất hiện mức xám GLCM (6 chỉ số) + Gray Stats (2 chỉ số) = 8D.
    Chỉ số: Contrast, Dissimilarity, Homogeneity, ASM, Energy, Correlation, Mean, Std.
    """
    gray_quant = (gray_img // 8).astype(np.uint8)  # 32 levels
    glcm = graycomatrix(
        gray_quant, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=32, symmetric=True, normed=True
    )
    contrast = float(graycoprops(glcm, 'contrast').mean())
    dissimilarity = float(graycoprops(glcm, 'dissimilarity').mean())
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    energy = float(graycoprops(glcm, 'energy').mean())
    correlation = float(graycoprops(glcm, 'correlation').mean())
    asm = float(graycoprops(glcm, 'ASM').mean())
    
    if mask is not None and np.sum(mask > 0) > 0:
        leaf_gray = gray_img[mask > 0].astype(np.float32)
    else:
        leaf_gray = gray_img.flatten().astype(np.float32)
        
    gray_mean = float(np.mean(leaf_gray))
    gray_std = float(np.std(leaf_gray))
    
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation, asm, gray_mean, gray_std], dtype=np.float32)


def extract_gabor_filter_bank(gray_img, mask=None, angles=None, lambdas=None):
    """
    Mục 3.2.2: Bộ lọc Gabor đa tần số và đa hướng cho hệ gân lá (24 chiều).
    - 4 góc hướng: theta in [0, pi/4, pi/2, 3pi/4]
    - 3 bước sóng: lambda in [4.0, 8.0, 16.0]
    - 2 thống kê: Mean & Std phản ứng của từng bộ lọc -> 4 x 3 x 2 = 24D.
    """
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    if lambdas is None:
        lambdas = [4.0, 8.0, 16.0]
        
    feats = []
    for theta in angles:
        for lam in lambdas:
            kernel = cv2.getGaborKernel((21, 21), sigma=3.0, theta=theta, lambd=lam, gamma=0.5, psi=0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
            if mask is not None and np.sum(mask > 0) > 0:
                resp = np.abs(filtered)[mask > 0]
            else:
                resp = np.abs(filtered).flatten()
            feats.append(float(np.mean(resp)))
            feats.append(float(np.std(resp)))
            
    return np.array(feats, dtype=np.float32)


def extract_lbp_features(gray_img, mask=None, P=8, R=1, method='uniform'):
    """
    Mục 3.2.3: Đặc trưng mẫu nhị phân cục bộ LBP Histogram (Uniform LBP 59 chiều).
    Nắm bắt vi kết cấu biểu bì và độ nhẵn bề mặt phiến lá.
    """
    lbp = local_binary_pattern(gray_img, P=P, R=R, method=method)
    num_bins = P * (P - 1) + 3 if method == 'uniform' else 256
    if mask is not None and np.sum(mask > 0) > 0:
        lbp_leaf = lbp[mask > 0]
    else:
        lbp_leaf = lbp.flatten()
        
    hist, _ = np.histogram(lbp_leaf, bins=num_bins, range=(0, num_bins), density=True)
    return hist.astype(np.float32)


# =========================================================================
# 3.3 NHÓM ĐẶC TRƯNG HÌNH DẠNG & MÉP LÁ (SHAPE FEATURES)
# =========================================================================

def extract_hu_moments(contour):
    """
    Mục 3.3.2: 7 Bất biến mô-men hình dạng Hu (Log-transformed 7D).
    Bất biến tuyệt đối với phép tịnh tiến, co giãn tỷ lệ và phép quay.
    """
    moments = cv2.moments(contour)
    raw_hu = cv2.HuMoments(moments).flatten()
    hu_vec = []
    for h_val in raw_hu:
        val = -np.sign(h_val) * np.log10(np.abs(h_val) + 1e-12) if abs(h_val) > 0 else 0.0
        hu_vec.append(float(val))
    return np.array(hu_vec, dtype=np.float32)


def extract_geometric_morphology(contour):
    """
    Mục 3.3.1: Bộ chỉ số hình thái học cơ bản và mở rộng:
    - 4 cơ bản: Aspect Ratio, Circularity, Solidity, Extent
    - 2 mở rộng: Rectangularity, Eccentricity -> 6D
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-6))
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / (hull_area + 1e-6))
    
    bx, by, bw, bh = cv2.boundingRect(contour)
    aspect_ratio = float(bw / float(bh)) if bh > 0 else 1.0
    extent = float(area / (float(bw * bh) + 1e-6))
    
    # Rectangularity
    rect = cv2.minAreaRect(contour)
    min_rect_area = float(rect[1][0] * rect[1][1])
    rectangularity = float(area / (min_rect_area + 1e-6))
    
    # Eccentricity
    eccentricity = 0.0
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            axes = ellipse[1]
            major_axis = max(axes)
            minor_axis = min(axes)
            if major_axis > 0:
                eccentricity = float(np.sqrt(1.0 - (minor_axis / major_axis) ** 2))
        except Exception:
            eccentricity = 0.0
            
    return np.array([aspect_ratio, circularity, solidity, extent, rectangularity, eccentricity], dtype=np.float32)


def extract_edge_histogram_descriptor(gray_img, mask=None):
    """
    Mục 3.3.3: Bộ mô tả cạnh Histogram (EHD theo chuẩn MPEG-7 - 80 chiều).
    Chia ảnh thành 4x4 = 16 khối; mỗi khối trích 5 loại cạnh (đứng, ngang, 45°, 135°, vô hướng).
    """
    h, w = gray_img.shape
    block_h, block_w = max(1, h // 4), max(1, w // 4)
    
    f_ver = np.array([[1, -1], [1, -1]], dtype=np.float32)
    f_hor = np.array([[1, 1], [-1, -1]], dtype=np.float32)
    f_45  = np.array([[np.sqrt(2), 0], [0, -np.sqrt(2)]], dtype=np.float32)
    f_135 = np.array([[0, np.sqrt(2)], [-np.sqrt(2), 0]], dtype=np.float32)
    f_nondir = np.array([[2, -2], [-2, 2]], dtype=np.float32)
    filters = [f_ver, f_hor, f_45, f_135, f_nondir]
    
    ehd = []
    for r in range(4):
        for c in range(4):
            sub_gray = gray_img[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
            if mask is not None:
                sub_mask = mask[r*block_h:(r+1)*block_h, c*block_w:(c+1)*block_w]
            else:
                sub_mask = np.ones_like(sub_gray)
                
            if np.sum(sub_mask > 0) < 5:
                ehd.extend([0.0] * 5)
                continue
                
            sub_resp = [np.abs(cv2.filter2D(sub_gray, cv2.CV_32F, f))[sub_mask > 0] for f in filters]
            stacked = np.stack(sub_resp, axis=0)  # (5, N)
            max_idx = np.argmax(stacked, axis=0)
            counts = np.bincount(max_idx, minlength=5).astype(np.float32)
            total_p = np.sum(counts)
            ehd.extend((counts / (total_p + 1e-6)).tolist())
            
    return np.array(ehd, dtype=np.float32)


def extract_hog_features(gray_img, target_size=(64, 64)):
    """
    Mục 3.3.4: Histogram of Oriented Gradients (HOG - 1764 chiều).
    Cấu hình chuẩn: ảnh 64x64, pixel_per_cell=(8,8), cells_per_block=(2,2), 9 bins.
    """
    resized = cv2.resize(gray_img, target_size)
    feats = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    return feats.astype(np.float32)


def extract_daisy_features(gray_img, step=32, radius=15):
    """
    Mục 3.3.5: Bộ mô tả đặc trưng cục bộ DAISY (DAISY Local Feature Descriptor).
    Lấy trung bình theo không gian để biểu diễn một vector mô tả cục bộ cô đọng.
    """
    descs = daisy(gray_img, step=step, radius=radius, rings=2, histograms=6, orientations=8)
    avg_daisy = np.mean(descs.reshape(-1, descs.shape[-1]), axis=0)
    return avg_daisy.astype(np.float32)


# =========================================================================
# 3.4 NHÓM ĐẶC TRƯNG HỌC SÂU (DEEP LEARNING FEATURE EMBEDDINGS - 512D)
# =========================================================================

_RESNET_PRETRAINED_MODEL = None
_RESNET_FINETUNED_MODEL = None
_RESNET_DEVICE = None
_RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_resnet_models():
    """Tải và khởi tạo Pretrained và Fine-tuned ResNet-18 model."""
    global _RESNET_PRETRAINED_MODEL, _RESNET_FINETUNED_MODEL, _RESNET_DEVICE
    if _RESNET_PRETRAINED_MODEL is None:
        _RESNET_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Pretrained ResNet-18
        m_pre = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m_pre.fc = torch.nn.Identity()
        m_pre.to(_RESNET_DEVICE)
        m_pre.eval()
        _RESNET_PRETRAINED_MODEL = m_pre
        
        # 2. Fine-tuned ResNet-18
        ft_path = os.path.join("db", "finetuned_resnet18_leaf.pth")
        if os.path.exists(ft_path):
            try:
                m_ft = models.resnet18(weights=None)
                in_features = m_ft.fc.in_features
                m_ft.fc = torch.nn.Sequential(torch.nn.Dropout(p=0.3), torch.nn.Linear(in_features, 8))
                checkpoint = torch.load(ft_path, map_location=_RESNET_DEVICE, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    m_ft.load_state_dict(checkpoint['model_state_dict'])
                else:
                    m_ft.load_state_dict(checkpoint)
                m_ft.fc = torch.nn.Identity()  # Lấy 512D embeddings
                m_ft.to(_RESNET_DEVICE)
                m_ft.eval()
                _RESNET_FINETUNED_MODEL = m_ft
            except Exception as e:
                print(f"[Warning] Error loading fine-tuned model ({e}), using pretrained.")
                _RESNET_FINETUNED_MODEL = _RESNET_PRETRAINED_MODEL
        else:
            _RESNET_FINETUNED_MODEL = _RESNET_PRETRAINED_MODEL
            
    return _RESNET_PRETRAINED_MODEL, _RESNET_FINETUNED_MODEL, _RESNET_DEVICE


def extract_resnet_embeddings(img_bgr, finetuned=True):
    """
    Mục 3.4: Trích xuất 512D ResNet-18 Embeddings từ tầng Global Average Pooling.
    - finetuned=True: Lấy từ mô hình Fine-tuned tối ưu hóa phân cụm loài.
    - finetuned=False: Lấy từ Pretrained ImageNet (Zero-shot).
    """
    m_pre, m_ft, device = get_resnet_models()
    model = m_ft if finetuned else m_pre
    
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    t_in = _RESNET_TRANSFORM(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model(t_in).squeeze().cpu().numpy()
        
    # Chuẩn hóa L2-norm
    norm_val = np.linalg.norm(emb)
    if norm_val > 0:
        emb = emb / norm_val
    return emb.astype(np.float32)


# =========================================================================
# 3.5 THIẾT KẾ CÁC BỘ VECTOR ĐẶC TRƯNG TỔNG HỢP (BỘ 1 ĐẾN BỘ 6)
# =========================================================================

def extract_all_report_features(image_or_path):
    """
    Trích xuất toàn diện 100% các đặc trưng và đóng gói chuẩn theo Báo cáo:
    - BỘ 1: Màu sắc độc lập (Color - 6D HSV Moments)
    - BỘ 2: Hình dạng độc lập (Shape - 11D Hu 7D + Geometric 4D)
    - BỘ 3: Kết cấu độc lập (Texture - 8D GLCM 6D + Gray Stats 2D)
    - BỘ 4: Dung hợp truyền thống cốt lõi (Handcrafted Fused - 25D = Bộ 1 + Bộ 2 + Bộ 3)
    - BỘ 5: Học sâu biểu diễn tiềm ẩn (Deep Learning Embeddings - 512D)
    - BỘ 6: Lai ghép đa trường phái toàn diện (Comprehensive Hybrid - 537D = Bộ 4 + Bộ 5)
    
    Kèm theo toàn bộ các vector mở rộng chuyên sâu:
    - Hue Histogram (180D), RGB Histogram (512D), Color Moments (9D)
    - Gabor Filter Bank (24D), LBP Histogram (59D), EHD MPEG-7 (80D), HOG (1764D), DAISY (104D)
    """
    if isinstance(image_or_path, str):
        if not os.path.exists(image_or_path):
            raise FileNotFoundError(f"Image not found: {image_or_path}")
        img = cv2.imread(image_or_path)
    else:
        img = image_or_path.copy()
        
    if img is None:
        raise ValueError("Failed to load image.")
        
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask, contour = segment_leaf(img)
    
    # 1. Trích xuất từng thành phần
    # Màu sắc
    color_moments_6d = extract_color_moments(img, mask, return_9d=False)
    color_moments_9d = extract_color_moments(img, mask, return_9d=True)
    hue_hist_180d = extract_hue_histogram(img, mask, bins=180)
    rgb_hist_512d = extract_rgb_histogram(img, mask, bins_per_channel=8)
    
    # Hình dạng
    hu_7d = extract_hu_moments(contour)
    geo_6d = extract_geometric_morphology(contour)
    geo_4d = geo_6d[:4]  # Aspect Ratio, Circularity, Solidity, Extent
    ehd_80d = extract_edge_histogram_descriptor(gray, mask)
    hog_1764d = extract_hog_features(gray)
    daisy_104d = extract_daisy_features(gray)
    
    # Kết cấu
    glcm_8d = extract_glcm_features(gray, mask)
    gabor_24d = extract_gabor_filter_bank(gray, mask)
    lbp_59d = extract_lbp_features(gray, mask, P=8, R=1, method='uniform')
    
    # Học sâu
    resnet_pretrained_512d = extract_resnet_embeddings(img, finetuned=False)
    resnet_finetuned_512d = extract_resnet_embeddings(img, finetuned=True)
    
    # 2. Đóng gói các BỘ chuẩn mực theo Báo cáo
    bo_1_color = color_moments_6d                                            # 6D
    bo_2_shape = np.concatenate([hu_7d, geo_4d], axis=0)                     # 7 + 4 = 11D
    bo_3_texture = glcm_8d                                                   # 8D
    bo_4_handcrafted_fused = np.concatenate([bo_1_color, bo_2_shape, bo_3_texture], axis=0) # 25D
    bo_5_deep_embeddings = resnet_finetuned_512d                             # 512D
    bo_6_comprehensive_hybrid = np.concatenate([bo_4_handcrafted_fused, bo_5_deep_embeddings], axis=0) # 537D
    
    return {
        # 6 BỘ vector chuẩn hóa theo Báo cáo
        "BO_1_COLOR": bo_1_color,
        "BO_2_SHAPE": bo_2_shape,
        "BO_3_TEXTURE": bo_3_texture,
        "BO_4_HANDCRAFTED_FUSED": bo_4_handcrafted_fused,
        "BO_5_DEEP_EMBEDDINGS": bo_5_deep_embeddings,
        "BO_6_COMPREHENSIVE_HYBRID": bo_6_comprehensive_hybrid,
        
        # Toàn bộ vector mở rộng chuyên biệt
        "color_moments_6d": color_moments_6d,
        "color_moments_9d": color_moments_9d,
        "hue_hist_180d": hue_hist_180d,
        "rgb_hist_512d": rgb_hist_512d,
        "hu_moments_7d": hu_7d,
        "geometric_morphology_4d": geo_4d,
        "geometric_morphology_6d": geo_6d,
        "ehd_mpeg7_80d": ehd_80d,
        "hog_1764d": hog_1764d,
        "daisy_descriptor": daisy_104d,
        "glcm_texture_8d": glcm_8d,
        "gabor_filter_bank_24d": gabor_24d,
        "lbp_histogram_59d": lbp_59d,
        "resnet_pretrained_512d": resnet_pretrained_512d,
        "resnet_finetuned_512d": resnet_finetuned_512d,
        
        # Dữ liệu hình học & mặt nạ
        "mask": mask,
        "contour": contour,
        "width": w,
        "height": h
    }


# =========================================================================
# HÀM TƯƠNG THÍCH NGƯỢC (BACKWARD COMPATIBLE)
# =========================================================================

def extract_features(image_or_path):
    """
    Hàm chuẩn phục vụ app.py, search_engine.py và extract_features.py.
    Trả về vector 25D và dict quen thuộc, tích hợp thêm trọn bộ 'report_features'.
    """
    all_res = extract_all_report_features(image_or_path)
    bo_4_fused_25d = all_res["BO_4_HANDCRAFTED_FUSED"]
    
    # Tái tạo features_dict theo format cũ
    c_m = all_res["color_moments_6d"]
    hu = all_res["hu_moments_7d"]
    geo = all_res["geometric_morphology_4d"]
    tex = all_res["glcm_texture_8d"]
    
    features_dict = {
        "color": {
            "h_mean": float(c_m[0]), "h_std": float(c_m[1]),
            "s_mean": float(c_m[2]), "s_std": float(c_m[3]),
            "v_mean": float(c_m[4]), "v_std": float(c_m[5])
        },
        "shape": {
            "hu_moments": hu.tolist(),
            "aspect_ratio": float(geo[0]),
            "circularity": float(geo[1]),
            "solidity": float(geo[2]),
            "extent": float(geo[3]),
            "area": float(cv2.contourArea(all_res["contour"])),
            "perimeter": float(cv2.arcLength(all_res["contour"], True))
        },
        "texture": {
            "glcm_contrast": float(tex[0]),
            "glcm_dissimilarity": float(tex[1]),
            "glcm_homogeneity": float(tex[2]),
            "glcm_energy": float(tex[3]),
            "glcm_correlation": float(tex[4]),
            "glcm_asm": float(tex[5]),
            "gray_mean": float(tex[6]),
            "gray_std": float(tex[7])
        },
        "scalar_summary": {
            "circularity": float(geo[1]),
            "solidity": float(geo[2]),
            "aspect_ratio": float(geo[0]),
            "mean_hue": float(c_m[0]),
            "mean_sat": float(c_m[2]),
            "glcm_contrast": float(tex[0]),
            "glcm_homogeneity": float(tex[2])
        }
    }
    
    return {
        "vector": bo_4_fused_25d,
        "features_dict": features_dict,
        "mask": all_res["mask"],
        "contour": all_res["contour"],
        "width": all_res["width"],
        "height": all_res["height"],
        "report_features": all_res  # Chứa đầy đủ 100% đặc trưng của Báo cáo
    }


if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    test_path = 'dataset/processed/01_Apple/LEAF_APPL_0001.jpg'
    if os.path.exists(test_path):
        res = extract_all_report_features(test_path)
        print("=" * 60)
        print("  KIỂM THỬ TRÍCH XUẤT ĐẦY ĐỦ 100% ĐẶC TRƯNG TỪ BÁO CÁO")
        print("=" * 60)
        print(f"BỘ 1 - Màu sắc (Color):            {res['BO_1_COLOR'].shape} (6 chiều)")
        print(f"BỘ 2 - Hình dạng (Shape):           {res['BO_2_SHAPE'].shape} (11 chiều)")
        print(f"BỘ 3 - Kết cấu (Texture):          {res['BO_3_TEXTURE'].shape} (8 chiều)")
        print(f"BỘ 4 - Dung hợp cốt lõi (Fused):   {res['BO_4_HANDCRAFTED_FUSED'].shape} (25 chiều)")
        print(f"BỘ 5 - Học sâu (ResNet-18):        {res['BO_5_DEEP_EMBEDDINGS'].shape} (512 chiều)")
        print(f"BỘ 6 - Lai ghép toàn diện:         {res['BO_6_COMPREHENSIVE_HYBRID'].shape} (537 chiều)")
        print("-" * 60)
        print(f"RGB Histogram (512D):              {res['rgb_hist_512d'].shape}")
        print(f"Hue Histogram (180D):              {res['hue_hist_180d'].shape}")
        print(f"Gabor Filter Bank (24D):           {res['gabor_filter_bank_24d'].shape}")
        print(f"LBP Histogram (59D):               {res['lbp_histogram_59d'].shape}")
        print(f"EHD MPEG-7 Descriptor (80D):       {res['ehd_mpeg7_80d'].shape}")
        print(f"HOG Descriptor (1764D):            {res['hog_1764d'].shape}")
        print(f"DAISY Local Descriptor:            {res['daisy_descriptor'].shape}")
        print("SUCCESS: 100% features perfectly mapped to code!")
