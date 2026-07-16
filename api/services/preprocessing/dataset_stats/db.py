from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, String, ForeignKey, JSON, create_engine, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass

class Dataset(Base):
    __tablename__ = "dataset"
    __table_args__ = (UniqueConstraint("name","split"),)
    
    id: Mapped[int] = mapped_column(primary_key= True)
    name: Mapped[str]
    split: Mapped[str]
    root_path: Mapped[str]
    config_path: Mapped[Optional[str]]
    num_images: Mapped[int] = mapped_column(default=0)
    num_boxes: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    boxes: Mapped[list["Box"]] = relationship(back_populates="dataset")
    
class Box(Base):
    __tablename__ = "boxes"
    
    id: Mapped[int] = mapped_column(primary_key= True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    image_id: Mapped[str]
    image_height: Mapped[int]
    image_width: Mapped[int]
    box_width: Mapped[float]
    box_height: Mapped[float]
    norm_width: Mapped[float]
    norm_height: Mapped[float]
    class_label: Mapped[int]
    dataset: Mapped["Dataset"] = relationship(back_populates="boxes")
    
def build_engine(url:str):
    engine = create_engine(url=url)
    Base.metadata.create_all(engine)
    return engine
