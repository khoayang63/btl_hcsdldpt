"""
Test improved segmentation: Background subtraction from corners + LAB color distance
"""
import cv2
import numpy as np
import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def segment_leaf_improved(img):
    """
    Improved leaf segmentation using background color sampling from corners.
    Strategy: Sample background color from 4 corners -> compute color distance
    for each pixel -> threshold on distance -> morphological cleanup.
    """
    h, w = img.shape[:2]

    # 1. Convert to LAB color space (better perceptual color distance)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 2. Sample background from 4 corners (each corner = 5% of image)
    margin_h = max(int(h * 0.05), 5)
    margin_w = max(int(w * 0.05), 5)
    corners = [
        lab[0:margin_h, 0:margin_w],              # Top-left
        lab[0:margin_h, w-margin_w:w],             # Top-right
        lab[h-margin_h:h, 0:margin_w],             # Bottom-left
        lab[h-margin_h:h, w-margin_w:w],           # Bottom-right
    ]
    bg_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
    bg_mean = np.mean(bg_pixels, axis=0)  # Average background color in LAB

    # 3. Compute per-pixel color distance from background
    diff = lab - bg_mean.reshape(1, 1, 3)
    color_dist = np.sqrt(np.sum(diff ** 2, axis=2))  # Euclidean distance in LAB

    # 4. Otsu threshold on distance map (auto-find best cutoff)
    dist_norm = cv2.normalize(color_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, dist_mask = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Morphological cleanup: fill holes, remove small noise
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(dist_mask, cv2.MORPH_CLOSE, kernel_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

    # 6. Fill internal holes using flood fill from corners
    flood = cleaned.copy()
    mask_flood = np.zeros((h + 2, w + 2), np.uint8)
    # Flood fill background from all 4 corners
    cv2.floodFill(flood, mask_flood, (0, 0), 128)
    cv2.floodFill(flood, mask_flood, (w - 1, 0), 128)
    cv2.floodFill(flood, mask_flood, (0, h - 1), 128)
    cv2.floodFill(flood, mask_flood, (w - 1, h - 1), 128)
    # Pixels that weren't reached by flood fill = internal holes = should be leaf
    holes = (flood != 128) & (cleaned == 0)
    cleaned[holes] = 255

    # 7. Find largest contour
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Ultimate fallback
        cleaned = np.zeros((h, w), dtype=np.uint8)
        cleaned[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)] = 255
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = max(contours, key=cv2.contourArea)

    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(final_mask, [largest], -1, 255, thickness=cv2.FILLED)

    return final_mask, largest


def segment_leaf_old(img):
    """Original thresholding method for comparison."""
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
        h, w = img.shape[:2]
        cleaned = np.zeros((h, w), dtype=np.uint8)
        cleaned[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = 255
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(gray)
    cv2.drawContours(final_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return final_mask, largest


if __name__ == '__main__':
    with open('data/dataset_metadata.json', 'r') as f:
        meta = json.load(f)

    old_bad = 0
    new_bad = 0
    old_ratios = []
    new_ratios = []
    improvements = []

    for idx, item in enumerate(meta):
        img = cv2.imread(item['filepath'])
        if img is None:
            continue
        h, w = img.shape[:2]
        total = h * w

        _, c_old = segment_leaf_old(img)
        old_ratio = cv2.contourArea(c_old) / total * 100

        _, c_new = segment_leaf_improved(img)
        new_ratio = cv2.contourArea(c_new) / total * 100

        old_ratios.append(old_ratio)
        new_ratios.append(new_ratio)

        if old_ratio < 10:
            old_bad += 1
        if new_ratio < 10:
            new_bad += 1

        improvements.append(new_ratio - old_ratio)

        if (idx + 1) % 200 == 0:
            print(f"  Checked {idx+1}/1200...")

    print(f"\n{'='*60}")
    print(f"  COMPARISON: Old vs Improved Segmentation")
    print(f"{'='*60}")
    print(f"  Bad masks (leaf < 10%):")
    print(f"    Old method:      {old_bad:3d} / 1200  ({old_bad*100/1200:.1f}%)")
    print(f"    Improved method: {new_bad:3d} / 1200  ({new_bad*100/1200:.1f}%)")
    print(f"")
    print(f"  Average leaf area ratio:")
    print(f"    Old method:      {np.mean(old_ratios):.1f}%")
    print(f"    Improved method: {np.mean(new_ratios):.1f}%")
    print(f"")
    print(f"  Images improved:   {sum(1 for d in improvements if d > 5)}")
    print(f"  Images degraded:   {sum(1 for d in improvements if d < -5)}")
    print(f"{'='*60}")
