from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String, ForeignKey, JSON, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Video(Base):
    __tablename__ ="videos"
    
    id: Mapped[int] = mapped_column(primary_key= True)
    source_file: Mapped[str]
    s3_key: Mapped[Optional[str]]
    dataset_name: Mapped[Optional[str]] = mapped_column(String(100))
    filename: Mapped[str] = mapped_column(String(255))
    duration: Mapped[float]
    fps: Mapped[float]
    total_frames: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime())
    width: Mapped[int]
    height: Mapped[int]
    status: Mapped[str] = mapped_column(String(30), default="pending")
    frames: Mapped[list["Frame"]] = relationship(back_populates="video")
    
class Frame(Base):
    __tablename__="frames"
    
    id: Mapped[int] = mapped_column(primary_key= True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    frame_index: Mapped[int]
    timestamp_s: Mapped[float]
    scene_change_score: Mapped[float]
    width: Mapped[int]
    height: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=datetime.utcnow)
    video: Mapped["Video"] = relationship(back_populates="frames")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="frame")
    
    
class Annotation(Base):
    __tablename__="annotations"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    frame_id: Mapped[int] = mapped_column(ForeignKey("frames.id"))
    x1: Mapped[float]
    y1: Mapped[float]
    x2: Mapped[float]
    y2: Mapped[float]
    class_id: Mapped[int]
    class_name: Mapped[str]
    votes: Mapped[int]
    consensus_score: Mapped[float]
    model_confidences: Mapped[dict] = mapped_column(JSON())
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=datetime.utcnow)
    frame: Mapped["Frame"] = relationship(back_populates="annotations")
    
class ProcessingJob(Base):
    __tablename__="processing_jobs"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    worker_id: Mapped[Optional[str]] = mapped_column(String(50))
    started_at: Mapped[datetime] = mapped_column(DateTime(), default=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]]
    frames_processed: Mapped[int] = mapped_column(default=0)
    annotations_generated: Mapped[int] = mapped_column(default=0)
    error_message: Mapped[Optional[str]]
    
    
def build_engine(url: str):
    # Creating the engine for the database
    engine = create_engine(url= url)
    Base.metadata.create_all(engine)
    return engine
    
    