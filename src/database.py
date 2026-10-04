import sqlite3
import json
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db', 'leaf_database.db')

def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute('PRAGMA foreign_keys = ON;')
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT
        );
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        );
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS features (
            image_id INTEGER PRIMARY KEY,
            vector_json TEXT NOT NULL,
            vector_blob BLOB NOT NULL,
            circularity REAL,
            solidity REAL,
            aspect_ratio REAL,
            mean_hue REAL,
            mean_sat REAL,
            glcm_contrast REAL,
            glcm_homogeneity REAL,
            FOREIGN KEY (image_id) REFERENCES images(id)
        );
    """)
    
    conn.commit()
    conn.close()
    print(f"Initialized database schema at {db_path}")

def insert_category(name, description='', db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO categories (name, description) VALUES (?, ?)", (name, description))
    conn.commit()
    cur.execute("SELECT id FROM categories WHERE name = ?", (name,))
    row = cur.fetchone()
    cat_id = row['id'] if row else None
    conn.close()
    return cat_id

def insert_image_record(filename, filepath, category_name, width, height, feature_vector, feature_dict, db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT id FROM categories WHERE name = ?", (category_name,))
    row = cur.fetchone()
    if row:
        cat_id = row['id']
    else:
        cur.execute("INSERT INTO categories (name, description) VALUES (?, ?)", (category_name, f"Leaf category {category_name}"))
        cat_id = cur.lastrowid

    cur.execute("""
        INSERT OR REPLACE INTO images (filename, filepath, category_id, width, height)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, filepath, cat_id, width, height))
    image_id = cur.lastrowid

    vec_array = np.array(feature_vector, dtype=np.float32)
    vec_json = json.dumps([float(x) for x in feature_vector])
    vec_blob = vec_array.tobytes()

    cur.execute("""
        INSERT OR REPLACE INTO features (
            image_id, vector_json, vector_blob,
            circularity, solidity, aspect_ratio,
            mean_hue, mean_sat, glcm_contrast, glcm_homogeneity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        image_id, vec_json, vec_blob,
        float(feature_dict.get('circularity', 0.0)),
        float(feature_dict.get('solidity', 0.0)),
        float(feature_dict.get('aspect_ratio', 0.0)),
        float(feature_dict.get('mean_hue', 0.0)),
        float(feature_dict.get('mean_sat', 0.0)),
        float(feature_dict.get('glcm_contrast', 0.0)),
        float(feature_dict.get('glcm_homogeneity', 0.0))
    ))
    
    conn.commit()
    conn.close()
    return image_id

def get_all_features(db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT f.image_id, f.vector_blob, img.filename, img.filepath, cat.name as category_name
        FROM features f
        JOIN images img ON f.image_id = img.id
        JOIN categories cat ON img.category_id = cat.id
        ORDER BY f.image_id ASC
    """)
    rows = cur.fetchall()
    conn.close()
    
    image_ids = []
    filenames = []
    filepaths = []
    categories = []
    vectors = []
    
    for r in rows:
        image_ids.append(r['image_id'])
        filenames.append(r['filename'])
        filepaths.append(r['filepath'])
        categories.append(r['category_name'])
        vec = np.frombuffer(r['vector_blob'], dtype=np.float32)
        vectors.append(vec)
        
    if vectors:
        feature_matrix = np.vstack(vectors)
    else:
        feature_matrix = np.empty((0, 25), dtype=np.float32)
        
    return image_ids, filenames, filepaths, categories, feature_matrix

def get_stats(db_path=DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as total_images FROM images")
    total_images = cur.fetchone()['total_images']
    
    cur.execute("""
        SELECT cat.name, COUNT(img.id) as count
        FROM categories cat
        LEFT JOIN images img ON cat.id = img.category_id
        GROUP BY cat.id, cat.name
        ORDER BY count DESC
    """)
    cat_counts = {r['name']: r['count'] for r in cur.fetchall()}
    conn.close()
    return {
        'total_images': total_images,
        'categories': cat_counts
    }

if __name__ == '__main__':
    init_db()
