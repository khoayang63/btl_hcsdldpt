"""
extract_features.py - Trích xuất đặc trưng 25D cho toàn bộ ảnh và nạp CSDL SQLite
Chạy: venv/Scripts/python extract_features.py
"""
import os
import sys
import json
import time
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.feature_extractor import extract_features
from src import database

DATA_DIR = "data"
DB_DIR = "db"

def run_extraction():
    print("=" * 60)
    print("  TRÍCH RÚT ĐẶC TRƯNG & NẠP CSDL SQLITE (BiRefNet_lite)")
    print("=" * 60)

    os.makedirs(DB_DIR, exist_ok=True)
    db_path = os.path.join(DB_DIR, "leaf_database.db")
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"Cleared old database at {db_path}")
        except Exception:
            pass
    database.init_db(db_path)

    meta_path = os.path.join(DATA_DIR, "dataset_metadata.json")
    if not os.path.exists(meta_path):
        print(f"[ERROR] {meta_path} not found. Run setup_dataset.py first.")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    total = len(metadata)
    print(f"Total images: {total}\n")

    start = time.time()
    count = 0

    for idx, item in enumerate(metadata, 1):
        try:
            res = extract_features(item["filepath"])
            database.insert_image_record(
                filename=item["filename"],
                filepath=item["filepath"],
                category_name=item["category"],
                width=res["width"], height=res["height"],
                feature_vector=res["vector"],
                feature_dict=res["features_dict"]["scalar_summary"],
                db_path=db_path
            )
            count += 1
            if idx % 100 == 0 or idx == total:
                elapsed = time.time() - start
                speed = idx / elapsed
                remain = (total - idx) / speed if speed > 0 else 0
                print(f"  [{idx:4d}/{total}] {idx*100//total:3d}% | {speed:.1f} img/s | ~{remain:.0f}s left")
        except Exception as e:
            print(f"  [!] Error on {item['filename']}: {e}")

    elapsed = time.time() - start
    print(f"\nExtracted {count}/{total} images in {elapsed:.1f}s ({elapsed/max(count,1)*1000:.1f} ms/img)")

    # Build normalization cache
    print("\nBuilding feature index cache...")
    ids, fnames, fpaths, cats, raw = database.get_all_features(db_path)
    mean_v = np.mean(raw, axis=0)
    std_v = np.std(raw, axis=0)
    std_v[std_v < 1e-6] = 1.0
    norm = (raw - mean_v) / std_v

    cache_path = os.path.join(DB_DIR, "feature_cache.npz")
    np.savez_compressed(cache_path,
        image_ids=np.array(ids), filenames=np.array(fnames),
        filepaths=np.array(fpaths), categories=np.array(cats),
        raw_matrix=raw, norm_matrix=norm,
        mean_vec=mean_v, std_vec=std_v)
    print(f"Saved cache -> {cache_path}")

    stats = database.get_stats(db_path)
    print(f"\nDatabase: {stats['total_images']} images")
    for cat, cnt in stats["categories"].items():
        print(f"  {cat:15s}: {cnt} imgs")
    print("=" * 60)

if __name__ == '__main__':
    run_extraction()
