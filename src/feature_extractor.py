import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
import torch
from torchvision import transforms
from PIL import Image

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

# ─── BIREFNET MODEL SINGLETON ───
_BIREFNET_MODEL = None
_BIREFNET_DEVICE = None
_BIREFNET_TRANSFORM = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_birefnet_model():
    """Lazy load BiRefNet model once into GPU / CPU."""
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
    High-accuracy leaf segmentation using BiRefNet_lite AI model.
    Falls back to adaptive thresholding if model inference fails.
    Returns: binary_mask (uint8 0/255), largest_contour
    """
    h, w = img.shape[:2]
    
    try:
        model, device = get_birefnet_model()
        
        # Convert BGR (cv2) to RGB (PIL)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Inference
        input_tensor = _BIREFNET_TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()
            
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred).resize((w, h), Image.BILINEAR)
        pred_np = np.array(pred_pil)
        
        # Threshold at 0.5 (128) for clean binary mask
        binary_mask = (pred_np > 128).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and max(cv2.contourArea(c) for c in contours) > 100:
            largest_contour = max(contours, key=cv2.contourArea)
            # Refine mask to fill any internal micro-holes
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


def extract_features(image_or_path):
    """
    Extracts 25-dimensional feature vector (Color, Shape, Texture) from leaf image.
    Returns: dict with 'vector', 'features_dict', 'mask', 'contour'
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mask, contour = segment_leaf(img)
    leaf_pixels_hsv = hsv[mask > 0]
    leaf_pixels_gray = gray[mask > 0]
    
    if len(leaf_pixels_hsv) == 0:
        leaf_pixels_hsv = hsv.reshape(-1, 3)
        leaf_pixels_gray = gray.reshape(-1)

    # ---------------- 1. COLOR FEATURES (6D) ----------------
    h_mean = float(np.mean(leaf_pixels_hsv[:, 0]))
    h_std = float(np.std(leaf_pixels_hsv[:, 0]))
    s_mean = float(np.mean(leaf_pixels_hsv[:, 1]))
    s_std = float(np.std(leaf_pixels_hsv[:, 1]))
    v_mean = float(np.mean(leaf_pixels_hsv[:, 2]))
    v_std = float(np.std(leaf_pixels_hsv[:, 2]))
    
    color_vec = [h_mean, h_std, s_mean, s_std, v_mean, v_std]

    # ---------------- 2. SHAPE FEATURES (11D) ----------------
    moments = cv2.moments(contour)
    raw_hu = cv2.HuMoments(moments).flatten()
    # Log-transformed Hu moments for numerical stability
    hu_vec = []
    for h_val in raw_hu:
        val = -np.sign(h_val) * np.log10(np.abs(h_val) + 1e-12) if abs(h_val) > 0 else 0.0
        hu_vec.append(float(val))
        
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter + 1e-6))
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / (hull_area + 1e-6))
    
    bx, by, bw, bh = cv2.boundingRect(contour)
    aspect_ratio = float(bw / float(bh)) if bh > 0 else 1.0
    extent = float(area / (float(bw * bh) + 1e-6))
    
    shape_vec = hu_vec + [aspect_ratio, circularity, solidity, extent]

    # ---------------- 3. TEXTURE FEATURES (8D) ----------------
    # Quantize to 32 levels for fast and robust GLCM computation
    gray_quant = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(gray_quant, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
    
    glcm_contrast = float(graycoprops(glcm, 'contrast').mean())
    glcm_dissimilarity = float(graycoprops(glcm, 'dissimilarity').mean())
    glcm_homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    glcm_energy = float(graycoprops(glcm, 'energy').mean())
    glcm_correlation = float(graycoprops(glcm, 'correlation').mean())
    glcm_asm = float(graycoprops(glcm, 'ASM').mean())
    
    leaf_gray_mean = float(np.mean(leaf_pixels_gray))
    leaf_gray_std = float(np.std(leaf_pixels_gray))
    
    texture_vec = [
        glcm_contrast, glcm_dissimilarity, glcm_homogeneity,
        glcm_energy, glcm_correlation, glcm_asm,
        leaf_gray_mean, leaf_gray_std
    ]

    # ---------------- FULL 25D VECTOR ----------------
    full_vector = np.array(color_vec + shape_vec + texture_vec, dtype=np.float32)
    
    features_dict = {
        "color": {
            "h_mean": h_mean, "h_std": h_std,
            "s_mean": s_mean, "s_std": s_std,
            "v_mean": v_mean, "v_std": v_std
        },
        "shape": {
            "hu_moments": hu_vec,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "solidity": solidity,
            "extent": extent,
            "area": float(area),
            "perimeter": float(perimeter)
        },
        "texture": {
            "glcm_contrast": glcm_contrast,
            "glcm_dissimilarity": glcm_dissimilarity,
            "glcm_homogeneity": glcm_homogeneity,
            "glcm_energy": glcm_energy,
            "glcm_correlation": glcm_correlation,
            "glcm_asm": glcm_asm,
            "gray_mean": leaf_gray_mean,
            "gray_std": leaf_gray_std
        },
        "scalar_summary": {
            "circularity": circularity,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "mean_hue": h_mean,
            "mean_sat": s_mean,
            "glcm_contrast": glcm_contrast,
            "glcm_homogeneity": glcm_homogeneity
        }
    }
    
    return {
        "vector": full_vector,
        "features_dict": features_dict,
        "mask": mask,
        "contour": contour,
        "width": w,
        "height": h
    }

if __name__ == '__main__':
    # Test on one image
    test_files = [f for f in os.listdir('dataset') if f.endswith('.jpg')]
    if test_files:
        test_path = os.path.join('dataset', test_files[0])
        res = extract_features(test_path)
        print(f"Extracted 25D vector from {test_files[0]}:")
        print(f"Vector shape: {res['vector'].shape}")
        print(f"Sample values: {res['vector'][:5]}")
        print("Scalar summary:", res['features_dict']['scalar_summary'])
