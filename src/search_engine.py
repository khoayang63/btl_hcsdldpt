import os
import sys
import numpy as np
import cv2
import base64

# Ensure project root is in path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_extractor import extract_features, FEATURE_NAMES

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CACHE_FILE = os.path.join(PROJECT_ROOT, 'db', 'feature_cache.npz')

class LeafSearchEngine:
    def __init__(self, cache_file=CACHE_FILE):
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file {cache_file} not found. Run extract_features.py first.")
            
        data = np.load(cache_file, allow_pickle=True)
        self.image_ids = data["image_ids"]
        self.filenames = data["filenames"]
        self.filepaths = data["filepaths"]
        self.categories = data["categories"]
        self.raw_matrix = data["raw_matrix"]
        self.norm_matrix = data["norm_matrix"]
        self.mean_vec = data["mean_vec"]
        self.std_vec = data["std_vec"]
        
        # Slices for feature groups
        # Color: 0..6 (6 dims)
        # Shape: 6..17 (11 dims)
        # Texture: 17..25 (8 dims)
        self.idx_color = slice(0, 6)
        self.idx_shape = slice(6, 17)
        self.idx_texture = slice(17, 25)

    def search(self, query_img_or_path, top_k=5, weights=None):
        """
        Performs CBIR search for a query leaf image.
        Returns:
            results: list of top_k matched images with similarity scores and breakdown
            intermediate_info: extracted features, mask, contour visualization
        """
        if weights is None:
            # Shape is primary, Texture secondary, Color tertiary
            weights = {"shape": 0.50, "texture": 0.30, "color": 0.20}
            
        # 1. Feature extraction on query
        feat_res = extract_features(query_img_or_path)
        raw_q = feat_res["vector"]
        
        # 2. Z-Score normalization using DB parameters
        norm_q = (raw_q - self.mean_vec) / self.std_vec
        
        # 3. Compute group distances
        # Euclidean distances in normalized space
        q_color = norm_q[self.idx_color]
        db_color = self.norm_matrix[:, self.idx_color]
        dist_color = np.linalg.norm(db_color - q_color, axis=1) / np.sqrt(6.0)

        q_shape = norm_q[self.idx_shape]
        db_shape = self.norm_matrix[:, self.idx_shape]
        dist_shape = np.linalg.norm(db_shape - q_shape, axis=1) / np.sqrt(11.0)

        q_texture = norm_q[self.idx_texture]
        db_texture = self.norm_matrix[:, self.idx_texture]
        dist_texture = np.linalg.norm(db_texture - q_texture, axis=1) / np.sqrt(8.0)

        # 4. Weighted combined distance
        total_dist = (
            weights["shape"] * dist_shape +
            weights["texture"] * dist_texture +
            weights["color"] * dist_color
        )
        
        # Convert distance to similarity percentage [0%, 100%]
        sim_scores = 100.0 * np.exp(-total_dist / 1.5)
        sim_scores = np.clip(sim_scores, 0.0, 100.0)
        
        # 5. Rank top_k
        top_indices = np.argsort(total_dist)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "rank": rank,
                "image_id": int(self.image_ids[idx]),
                "filename": str(self.filenames[idx]),
                "filepath": str(self.filepaths[idx]).replace("\\", "/"),
                "category": str(self.categories[idx]),
                "similarity_pct": round(float(sim_scores[idx]), 2),
                "total_dist": round(float(total_dist[idx]), 4),
                "dist_shape": round(float(dist_shape[idx]), 4),
                "dist_texture": round(float(dist_texture[idx]), 4),
                "dist_color": round(float(dist_color[idx]), 4)
            })
            
        # ==================== VISUALIZATIONS ====================
        mask = feat_res["mask"]
        contour = feat_res["contour"]
        
        if isinstance(query_img_or_path, str):
            orig_bgr = cv2.imread(query_img_or_path)
        else:
            orig_bgr = query_img_or_path.copy()

        h_img, w_img = orig_bgr.shape[:2]

        # --- VIS 1: Contour overlay (green) ---
        vis_contour = orig_bgr.copy()
        cv2.drawContours(vis_contour, [contour], -1, (0, 255, 0), 2)

        # --- VIS 2: Masked crop (leaf only, black bg) ---
        vis_masked = cv2.bitwise_and(orig_bgr, orig_bgr, mask=mask)

        # --- VIS 3: Shape overlay (contour + convex hull + bounding rect) ---
        vis_shape = orig_bgr.copy()
        # Convex hull in red
        hull = cv2.convexHull(contour)
        cv2.drawContours(vis_shape, [hull], -1, (0, 0, 255), 2)
        # Contour in green
        cv2.drawContours(vis_shape, [contour], -1, (0, 255, 0), 2)
        # Bounding rect in cyan
        bx, by, bw, bh = cv2.boundingRect(contour)
        cv2.rectangle(vis_shape, (bx, by), (bx+bw, by+bh), (255, 255, 0), 2)
        # Fitted ellipse in magenta (if enough points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(vis_shape, ellipse, (255, 0, 255), 2)

        # --- VIS 4: Hue heatmap on leaf region ---
        hsv_img = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
        hue_ch = hsv_img[:, :, 0]
        hue_colored = cv2.applyColorMap((hue_ch * 1.4).clip(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
        vis_hue = cv2.bitwise_and(hue_colored, hue_colored, mask=mask)

        # --- VIS 5: HSV Histogram data (for Chart.js) ---
        leaf_hsv = hsv_img[mask > 0]
        hist_h = cv2.calcHist([hsv_img], [0], mask, [180], [0, 180]).flatten().tolist()
        hist_s = cv2.calcHist([hsv_img], [1], mask, [64], [0, 256]).flatten().tolist()
        hist_v = cv2.calcHist([hsv_img], [2], mask, [64], [0, 256]).flatten().tolist()

        # --- VIS 6: GLCM heatmap ---
        from skimage.feature import graycomatrix
        gray_full = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        gray_q = (gray_full // 8).astype(np.uint8)
        glcm_raw = graycomatrix(gray_q, distances=[1], angles=[0], levels=32, symmetric=True, normed=True)
        glcm_2d = glcm_raw[:, :, 0, 0]
        # Normalize for display
        glcm_vis = (glcm_2d / (glcm_2d.max() + 1e-10) * 255).astype(np.uint8)
        glcm_vis = cv2.resize(glcm_vis, (256, 256), interpolation=cv2.INTER_NEAREST)
        glcm_colored = cv2.applyColorMap(glcm_vis, cv2.COLORMAP_INFERNO)

        # --- VIS 7: Edge map (Canny) on leaf ---
        gray_leaf = cv2.bitwise_and(gray_full, gray_full, mask=mask)
        edges = cv2.Canny(gray_leaf, 50, 150)
        vis_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # --- Radar chart data (normalized 0-1 for display) ---
        feat_d = feat_res["features_dict"]
        radar_data = {
            "labels": ["Circularity", "Solidity", "Aspect Ratio", "Hue", "Saturation", "GLCM Contrast", "Homogeneity"],
            "values": [
                min(feat_d["shape"]["circularity"], 1.0),
                min(feat_d["shape"]["solidity"], 1.0),
                min(feat_d["shape"]["aspect_ratio"] / 2.0, 1.0),
                feat_d["color"]["h_mean"] / 180.0,
                feat_d["color"]["s_mean"] / 255.0,
                min(feat_d["texture"]["glcm_contrast"] / 10.0, 1.0),
                feat_d["texture"]["glcm_homogeneity"]
            ]
        }

        # --- Color swatch (average HSV -> BGR -> base64) ---
        avg_h = feat_d["color"]["h_mean"]
        avg_s = feat_d["color"]["s_mean"]
        avg_v = feat_d["color"]["v_mean"]
        swatch_hsv = np.array([[[int(avg_h), int(avg_s), int(avg_v)]]], dtype=np.uint8)
        swatch_bgr = cv2.cvtColor(swatch_hsv, cv2.COLOR_HSV2BGR)
        swatch_hex = "#{:02x}{:02x}{:02x}".format(int(swatch_bgr[0,0,2]), int(swatch_bgr[0,0,1]), int(swatch_bgr[0,0,0]))

        # === Encode all visualizations ===
        def encode_img(img_bgr):
            _, buf = cv2.imencode('.png', img_bgr)
            return base64.b64encode(buf).decode('utf-8')

        _, mask_buf = cv2.imencode('.png', mask)

        intermediate_info = {
            "features_dict": feat_res["features_dict"],
            "raw_vector": [round(float(x), 4) for x in raw_q],
            "norm_vector": [round(float(x), 4) for x in norm_q],
            "feature_names": FEATURE_NAMES,
            "mask_base64": base64.b64encode(mask_buf).decode('utf-8'),
            "contour_base64": encode_img(vis_contour),
            "masked_crop_base64": encode_img(vis_masked),
            "shape_overlay_base64": encode_img(vis_shape),
            "hue_heatmap_base64": encode_img(vis_hue),
            "glcm_heatmap_base64": encode_img(glcm_colored),
            "edge_map_base64": encode_img(vis_edges),
            "hist_h": hist_h,
            "hist_s": hist_s,
            "hist_v": hist_v,
            "radar": radar_data,
            "color_swatch_hex": swatch_hex
        }
        
        return results, intermediate_info


if __name__ == '__main__':
    engine = LeafSearchEngine()
    test_queries = [f for f in os.listdir('test_queries') if f.endswith('.jpg')]
    if test_queries:
        q_path = os.path.join('test_queries', test_queries[0])
        print(f"Testing search engine with query: {q_path}")
        results, inter = engine.search(q_path, top_k=5)
        print("\nTOP 5 KẾT QUẢ TƯƠNG ĐỒNG:")
        for r in results:
            print(f"  #{r['rank']}: {r['category']:12s} | Similarity: {r['similarity_pct']:6.2f}% | Dist: {r['total_dist']:.4f} | File: {r['filename']}")
