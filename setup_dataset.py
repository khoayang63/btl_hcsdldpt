"""
setup_dataset.py - Giải nén và chuẩn bị tập dữ liệu ảnh lá cây từ dataset.zip
Chạy: venv/Scripts/python setup_dataset.py
"""
import os
import sys
import zipfile
import json
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ZIP_PATH = "LEAF.v2i.coco.zip"
DATA_DIR = "data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
TEST_DIR = os.path.join(DATA_DIR, "test_queries")
TARGET_PER_CLASS = 150  # 150 x 8 = 1200 images

def setup():
    if not os.path.exists(ZIP_PATH):
        print(f"[ERROR] Archive not found: {ZIP_PATH}")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    print(f"[1/4] Reading archive: {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # Parse COCO annotations
        valid_coco = json.loads(z.read("valid/_annotations.coco.json").decode('utf-8'))
        cat_map = {c['id']: c['name'] for c in valid_coco['categories']}
        img_id_to_file = {img['id']: img['file_name'] for img in valid_coco['images']}
        img_id_to_cat = {}
        for ann in valid_coco['annotations']:
            img_id_to_cat[ann['image_id']] = cat_map.get(ann['category_id'], 'Leaf')

        cat_to_images = defaultdict(list)
        for img_id, cat_name in img_id_to_cat.items():
            fname = img_id_to_file.get(img_id)
            if fname and ("valid/" + fname) in z.namelist():
                cat_to_images[cat_name].append("valid/" + fname)

        print("[2/4] Images per category in valid split:")
        for cat, imgs in sorted(cat_to_images.items()):
            print(f"  - {cat}: {len(imgs)} images")

        # Extract dataset images
        print(f"\n[3/4] Extracting {TARGET_PER_CLASS} images/category -> {DATASET_DIR}/")
        extracted_meta = []
        for cat, imgs in cat_to_images.items():
            for arc_path in imgs[:TARGET_PER_CLASS]:
                base_name = os.path.basename(arc_path)
                out_path = os.path.join(DATASET_DIR, base_name)
                with open(out_path, "wb") as f_out:
                    f_out.write(z.read(arc_path))
                extracted_meta.append({"filename": base_name, "category": cat, "filepath": out_path})

        print(f"  -> {len(extracted_meta)} images extracted.")

        # Extract test query images
        print(f"\n[4/4] Extracting test queries -> {TEST_DIR}/")
        test_coco = json.loads(z.read("test/_annotations.coco.json").decode('utf-8'))
        test_cat_map = {c['id']: c['name'] for c in test_coco['categories']}
        test_img_to_file = {img['id']: img['file_name'] for img in test_coco['images']}
        test_img_to_cat = {}
        for ann in test_coco['annotations']:
            test_img_to_cat[ann['image_id']] = test_cat_map.get(ann['category_id'], 'Leaf')

        test_by_cat = defaultdict(list)
        for img_id, cat_name in test_img_to_cat.items():
            fname = test_img_to_file.get(img_id)
            if fname and ("test/" + fname) in z.namelist():
                test_by_cat[cat_name].append("test/" + fname)

        query_meta = []
        for cat, imgs in test_by_cat.items():
            for arc_path in imgs[:5]:
                base_name = "query_" + os.path.basename(arc_path)
                out_path = os.path.join(TEST_DIR, base_name)
                with open(out_path, "wb") as f_out:
                    f_out.write(z.read(arc_path))
                query_meta.append({"filename": base_name, "category": cat, "filepath": out_path})

        # Save metadata
        with open(os.path.join(DATA_DIR, "dataset_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(extracted_meta, f, indent=2, ensure_ascii=False)
        with open(os.path.join(DATA_DIR, "test_queries_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(query_meta, f, indent=2, ensure_ascii=False)

        print(f"  -> {len(query_meta)} test queries extracted.")
        print("\n[DONE] Setup completed!")

if __name__ == "__main__":
    setup()
