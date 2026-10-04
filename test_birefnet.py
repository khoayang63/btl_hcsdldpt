"""
test_birefnet.py - Chạy mô hình BiRefNet_lite trên 10 ảnh và xuất ra thư mục test_binary_mask
"""
import os
import sys
import json
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from transformers import AutoModelForImageSegmentation

OUTPUT_DIR = "test_binary_mask"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

print("Loading BiRefNet_lite model from Hugging Face...")
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet_lite", 
    trust_remote_code=True
)
birefnet.to(device)
birefnet.eval()
print("Model loaded successfully!")

# Image transform standard for BiRefNet (1024x1024 input)
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Pick 10 diverse images from test queries across categories
with open("data/test_queries_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# Pick 1 image from each of the 8 categories + 2 extra
selected_items = []
seen_cats = set()
for item in meta:
    cat = item["category"]
    if cat not in seen_cats:
        selected_items.append(item)
        seen_cats.add(cat)
    if len(selected_items) == 8:
        break

# Add 2 more distinct images
for item in meta:
    if item not in selected_items:
        selected_items.append(item)
    if len(selected_items) == 10:
        break

print(f"\nProcessing {len(selected_items)} images...")

for idx, item in enumerate(selected_items, 1):
    img_path = item["filepath"]
    cat = item["category"]
    fname = item["filename"]
    base_name = f"sample_{idx:02d}_{cat.lower()}"
    
    print(f"[{idx}/10] {cat}: {fname} ...", end=" ", flush=True)
    
    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    
    # 1. Save original image to test_binary_mask
    orig_save_path = os.path.join(OUTPUT_DIR, f"{base_name}_orig.jpg")
    image.save(orig_save_path)
    
    # 2. Run BiRefNet inference
    input_tensor = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()
    
    # 3. Postprocess mask back to original resolution
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil = pred_pil.resize((orig_w, orig_h), Image.BILINEAR)
    
    # Soft/Matte mask
    matte_path = os.path.join(OUTPUT_DIR, f"{base_name}_matte.png")
    pred_pil.save(matte_path)
    
    # Binary mask (threshold 0.5)
    pred_np = np.array(pred_pil)
    binary_mask = (pred_np > 128).astype(np.uint8) * 255
    binary_pil = Image.fromarray(binary_mask)
    binary_path = os.path.join(OUTPUT_DIR, f"{base_name}_mask.png")
    binary_pil.save(binary_path)
    
    # 4. Save transparent RGBA crop
    rgba_img = image.copy()
    rgba_img.putalpha(pred_pil)
    rgba_path = os.path.join(OUTPUT_DIR, f"{base_name}_cutout.png")
    rgba_img.save(rgba_path)
    
    print("Done!")

print(f"\nAll 10 samples processed! Output files saved in '{OUTPUT_DIR}/'")
