from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from ..config import ETL_DB_URL

router = APIRouter()

@router.get("/stats")
def get_stats():
    engine= create_engine(url=ETL_DB_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM videos"))
        total_videos = result.scalar()
        
        result = conn.execute(text("SELECT COUNT(*) FROM frames"))
        total_frames = result.scalar()
        
        result = conn.execute(text("SELECT COUNT(*) FROM annotations"))
        total_annotations = result.scalar()
        
        class_distributions = class_distributions = [
            {'class_name': row[0], 'count': row[1]}
            for row in conn.execute(text("SELECT class_name, COUNT(*) as count FROM annotations GROUP BY class_name ORDER BY count DESC")).all()
        ]
        
    return {
        'total_videos': total_videos,
        'total_frames': total_frames,
        'total_annotations': total_annotations,
        'class_distribution': class_distributions
    }
    
@router.get("/videos")
def get_videos():
    engine = create_engine(ETL_DB_URL)
    with engine.connect() as conn:
        query = text("""
            SELECT v.id, v.filename, v.duration, v.fps, v.width, v.height, v.status,
            COUNT(DISTINCT f.id) AS frames,
            COUNT(a.id) AS annotations
            FROM videos v
            LEFT JOIN frames f ON f.video_id = v.id
            LEFT JOIN annotations a ON a.frame_id = f.id
            GROUP BY v.id ORDER BY v.created_at DESC
        """)
        results = conn.execute(query).all()
        
    return [dict(row._mapping) for row in results]

@router.get("/videos/{video_id}/frames")
def get_frames(video_id: int):
    engine = create_engine(ETL_DB_URL)
    with engine.connect() as conn:
        query = text("""
            SELECT f.id, f.frame_index, f.timestamp_s, f.scene_change_score,
                   COUNT(a.id) AS annotation_count
            FROM frames f
            LEFT JOIN annotations a ON a.frame_id = f.id
            WHERE f.video_id = :vid
            GROUP BY f.id
            ORDER BY f.frame_index
        """)
        results = conn.execute(query, {"vid": video_id}).all()
    return [dict(row._mapping) for row in results]

@router.get("/frames/{frame_id}/annotations")
def get_annotations(frame_id: int):
    engine = create_engine(ETL_DB_URL)
    with engine.connect() as conn:
        query = text("""
            SELECT id, class_name, class_id, votes, consensus_score,
                   x1, y1, x2, y2, model_confidences
            FROM annotations
            WHERE frame_id = :fid
            ORDER BY id
        """)
        results = conn.execute(query, {"fid": frame_id}).all()
    return [dict(row._mapping) for row in results]