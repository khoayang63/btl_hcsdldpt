"""
src/cnn_extractor.py - Trích xuất Deep Feature Embeddings bằng ResNet-18 và Giảm chiều PCA / K-Means
"""
import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_RESNET_MODEL = None
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_resnet_feature_extractor():
    """
    Tải ResNet-18 pretrained và bỏ lớp phân loại cuối (fc),
    lấy vector embedding 512 chiều từ tầng AdaptiveAvgPool2d.
    """
    global _RESNET_MODEL
    if _RESNET_MODEL is None:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Bỏ fully connected layer, giữ lại feature extractor (512-dim)
        model.fc = nn.Identity()
        model.to(DEVICE)
        model.eval()
        _RESNET_MODEL = model
    return _RESNET_MODEL

def extract_cnn_embedding(image_or_path):
    """
    Trích xuất vector đặc trưng 512D từ một ảnh lá (path, PIL hoặc ndarray).
    Trả về numpy array 512D đã chuẩn hóa L2 (norm = 1.0).
    """
    model = get_resnet_feature_extractor()
    
    if isinstance(image_or_path, str):
        pil_img = Image.open(image_or_path).convert("RGB")
    elif isinstance(image_or_path, np.ndarray):
        import cv2
        rgb = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    elif isinstance(image_or_path, Image.Image):
        pil_img = image_or_path.convert("RGB")
    else:
        raise ValueError("Unsupported image type")

    tensor = _TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(tensor).cpu().squeeze().numpy()

    # L2 Normalization để so sánh khoảng cách Cosine / Euclidean tốt nhất
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm

    return embedding.astype(np.float32)

def extract_cnn_batch(image_paths, batch_size=32):
    """
    Trích xuất embedding cho danh sách ảnh theo batch để tối ưu tốc độ GPU.
    """
    model = get_resnet_feature_extractor()
    embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            tensors.append(_TRANSFORM(img))
        batch_tensor = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            batch_emb = model(batch_tensor).cpu().numpy()
            
        # L2 normalize
        norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        batch_emb = batch_emb / norms
        embeddings.append(batch_emb)
        
    return np.vstack(embeddings).astype(np.float32)
