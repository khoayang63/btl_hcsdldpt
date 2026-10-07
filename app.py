"""
app.py - Flask Web Demo: Hệ thống tìm kiếm ảnh lá cây tương đồng (CBIR)
Chạy: venv/Scripts/python app.py
Truy cập: http://localhost:5000
"""
import os
import sys
import json
import base64
import cv2
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from flask import Flask, render_template, request, jsonify, send_from_directory
from src.search_engine import LeafSearchEngine

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize search engine
engine = LeafSearchEngine()

@app.route('/')
def index():
    # Get sample test query images for quick demo
    test_dir = os.path.join(DATA_DIR, 'test_queries')
    samples = []
    if os.path.exists(test_dir):
        meta_path = os.path.join(DATA_DIR, 'test_queries_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            for item in meta:
                samples.append({
                    'filename': item['filename'],
                    'category': item['category']
                })
    return render_template('index.html', samples=samples)


@app.route('/search', methods=['POST'])
def search():
    query_path = None

    # Case 1: Upload file
    if 'query_image' in request.files and request.files['query_image'].filename:
        file = request.files['query_image']
        save_path = os.path.join(UPLOAD_DIR, 'query_upload.jpg')
        file.save(save_path)
        query_path = save_path

    # Case 2: Select sample
    elif 'sample_name' in request.form and request.form['sample_name']:
        sample_name = request.form['sample_name']
        query_path = os.path.join(DATA_DIR, 'test_queries', sample_name)

    if not query_path or not os.path.exists(query_path):
        return jsonify({'error': 'No valid query image provided.'}), 400

    # Parse search weights / mode
    mode = request.form.get('search_mode', 'balanced')
    weights = None
    if mode == 'shape_only':
        weights = {"shape": 1.0, "texture": 0.0, "color": 0.0}
    elif mode == 'color_only':
        weights = {"shape": 0.0, "texture": 0.0, "color": 1.0}
    elif mode == 'texture_only':
        weights = {"shape": 0.0, "texture": 1.0, "color": 0.0}
    elif mode == 'shape_texture':
        weights = {"shape": 0.65, "texture": 0.35, "color": 0.0}
    elif mode == 'custom':
        try:
            w_shape = float(request.form.get('w_shape', 0.5))
            w_texture = float(request.form.get('w_texture', 0.3))
            w_color = float(request.form.get('w_color', 0.2))
            total_w = w_shape + w_texture + w_color
            if total_w > 0:
                weights = {
                    "shape": w_shape / total_w,
                    "texture": w_texture / total_w,
                    "color": w_color / total_w
                }
        except (ValueError, TypeError):
            weights = {"shape": 0.50, "texture": 0.30, "color": 0.20}
    else:  # balanced
        weights = {"shape": 0.50, "texture": 0.30, "color": 0.20}

    # Run search
    if mode == 'finetuned_cnn':
        results, intermediate = engine.search_finetuned(query_path, top_k=5)
        intermediate['active_weights'] = None
        intermediate['search_mode'] = 'finetuned_cnn'
    elif mode == 'deep_cnn':
        results, intermediate = engine.search_cnn(query_path, top_k=5)
        intermediate['active_weights'] = None
        intermediate['search_mode'] = 'deep_cnn'
    else:
        results, intermediate = engine.search(query_path, top_k=5, weights=weights)
        intermediate['active_weights'] = weights
        intermediate['search_mode'] = mode

    # Encode query image as base64 for display
    with open(query_path, 'rb') as f:
        query_b64 = base64.b64encode(f.read()).decode('utf-8')

    # Encode result images as base64
    for r in results:
        img_path = r['filepath']
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                r['image_base64'] = base64.b64encode(f.read()).decode('utf-8')
        else:
            r['image_base64'] = ''

    return jsonify({
        'query_image': query_b64,
        'results': results,
        'intermediate': intermediate
    })


@app.route('/data/test_queries/<path:filename>')
def serve_test_query(filename):
    return send_from_directory(os.path.join(DATA_DIR, 'test_queries'), filename)


@app.route('/data/dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory(os.path.join(DATA_DIR, 'dataset'), filename)


@app.route('/visualization/<path:filename>')
def serve_visualization(filename):
    viz_dir = os.path.join(PROJECT_ROOT, 'visualization')
    return send_from_directory(viz_dir, filename)


@app.route('/compare', methods=['POST'])
def compare():
    """Extract features from a result image for side-by-side comparison."""
    from src.feature_extractor import extract_features
    filepath = request.json.get('filepath', '')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Image not found'}), 400

    try:
        res = extract_features(filepath)
        feat = res['features_dict']

        # Build same radar data structure as query
        radar_values = [
            min(feat["shape"]["circularity"], 1.0),
            min(feat["shape"]["solidity"], 1.0),
            min(feat["shape"]["aspect_ratio"] / 2.0, 1.0),
            feat["color"]["h_mean"] / 180.0,
            feat["color"]["s_mean"] / 255.0,
            min(feat["texture"]["glcm_contrast"] / 10.0, 1.0),
            feat["texture"]["glcm_homogeneity"]
        ]

        # Color swatch
        avg_h, avg_s, avg_v = int(feat["color"]["h_mean"]), int(feat["color"]["s_mean"]), int(feat["color"]["v_mean"])
        swatch_hsv = np.array([[[avg_h, avg_s, avg_v]]], dtype=np.uint8)
        swatch_bgr = cv2.cvtColor(swatch_hsv, cv2.COLOR_HSV2BGR)
        swatch_hex = "#{:02x}{:02x}{:02x}".format(int(swatch_bgr[0,0,2]), int(swatch_bgr[0,0,1]), int(swatch_bgr[0,0,0]))

        # Encode shape overlay
        img = cv2.imread(filepath)
        mask = res['mask']
        contour = res['contour']

        vis_shape = img.copy()
        hull = cv2.convexHull(contour)
        cv2.drawContours(vis_shape, [hull], -1, (0, 0, 255), 2)
        cv2.drawContours(vis_shape, [contour], -1, (0, 255, 0), 2)
        bx, by, bw, bh = cv2.boundingRect(contour)
        cv2.rectangle(vis_shape, (bx, by), (bx+bw, by+bh), (255, 255, 0), 2)
        if len(contour) >= 5:
            cv2.ellipse(vis_shape, cv2.fitEllipse(contour), (255, 0, 255), 2)
        _, shape_buf = cv2.imencode('.png', vis_shape)

        return jsonify({
            'features': feat,
            'radar_values': radar_values,
            'color_swatch_hex': swatch_hex,
            'raw_vector': [round(float(x), 4) for x in res['vector']],
            'shape_overlay_base64': base64.b64encode(shape_buf).decode('utf-8')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("  LEAF CBIR SYSTEM - Web Demo")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
