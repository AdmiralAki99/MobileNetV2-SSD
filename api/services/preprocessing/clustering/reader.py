import numpy as np
from sqlalchemy import text
from ..dataset_stats.db import build_engine

def read_box_dims(db_url: str, dataset_name: str, split: str):
    engine = build_engine(db_url)
    with engine.connect() as connection:
        rows = connection.execute(text("""
            SELECT b.box_width, b.box_height,
                   b.norm_width, b.norm_height,
                   b.image_width, b.image_height,
                   b.class_label
            FROM boxes b JOIN dataset d ON b.dataset_id = d.id
            WHERE d.name = :name AND d.split = :split
        """), {"name": dataset_name, "split": split}).fetchall()
        arr = np.array(rows, dtype=np.float64)
        
        return {
            'raw': arr[:, 0:2],
            'norm': arr[:, 2:4],
            "image":  arr[:, 4:6],
            "labels": arr[:, 6].astype(int),
        }
        