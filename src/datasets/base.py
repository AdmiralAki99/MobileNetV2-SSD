import numpy as np
from typing import Any, Iterator
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

@dataclass
class DetectionSample:
    image: np.ndarray  # Shape: [H, W, 3]
    boxes: np.ndarray  # Shape: [N, 4]
    labels: np.ndarray # Shape: [N]
    image_id: str
    path: str
    orig_size: tuple[int,int] # (height, width)
    
    def validate(self):
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"Image must be [H,W,3], got {self.image.shape}")

        if self.image.dtype != np.float32:
            raise ValueError(f"Image must be float32, got {self.image.dtype}")

        # Checking for the Boxes
        if self.boxes.ndim != 2 or (len(self.boxes) > 0 and self.boxes.shape[1] != 4):
            raise ValueError(f"Boxes must be [N,4], got {self.boxes.shape}")

        if self.boxes.dtype != np.float32:
            raise ValueError(f"Boxes must be float32, got {self.boxes.dtype}")

        # Checking for the Labels
        if self.labels.ndim != 1:
            raise ValueError(f"Labels must be [N], got {self.labels.shape}")

        if len(self.boxes) != len(self.labels):
            raise ValueError(f"Boxes ({len(self.boxes)}) and labels ({len(self.labels)}) count mismatch")

        if self.labels.dtype != np.int32:
            raise ValueError(f"Labels must be int32, got {self.labels.dtype}")

        if len(self.labels) > 0 and np.any(self.labels < 1):
            raise ValueError(f"Labels must be >= 1 (0 is background), got min={self.labels.min()}")


    def to_dict(self):
        return {
            "image": self.image,
            "boxes": self.boxes,
            "labels": self.labels,
            "image_id": self.image_id,
            "path": self.path,
            "orig_size": np.array(self.orig_size, dtype=np.int32),
        }
        
class BaseDetectionDataset(ABC):
    def __init__(self,root: str | Path, split: str, classes_file: str | Path, use_difficult: bool = False, validate: bool = True):
        
        self.root = Path(root)
        self.split = split
        self.use_difficult = use_difficult
        self._validate = validate

        self._class_names = self._load_classes(classes_file)

        self._class_to_index = {name: i + 1 for i, name in enumerate(self._class_names)}
        self._index_to_class = {i + 1: name for i, name in enumerate(self._class_names)}
        self._index_to_class[0] = "background"

    def _load_classes(self,classes_file: str | Path):
        classes_file = Path(classes_file)

        # Checking if it exists
        if not classes_file.exists():
            raise FileNotFoundError(f"Classes file not found: {classes_file}")

        with open(classes_file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_to_index(self):
        return self._class_to_index

    @property
    def class_to_index(self):
        return {name: i + 1 for i, name in enumerate(self.class_names)}

    @property
    def index_to_class(self):
        mapping = {i + 1: name for i, name in enumerate(self.class_names)}
        mapping[0] = "background"
        return mapping

    @property
    def index_to_class(self):
        return self._index_to_class

    @property
    def num_classes(self):
        return len(self._class_names) + 1

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def _load_sample(self, index: int):
        raise NotImplementedError

    def _clean_boxes(self, sample: DetectionSample):

        boxes = sample.boxes
        labels = sample.labels
        H,W = sample.image.shape[:2]

        # Checking for the boxes
        if len(boxes) == 0:
            return sample

        # Cleaning the boxes first
        boxes = boxes.copy()
        boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, W)
        boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, H)

        # Checking for the degenerate boxes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid_mask = (widths > 0) & (heights > 0)

        # Removing NaN Boxes
        finite_mask = np.all(np.isfinite(boxes), axis = 1)
        valid_mask = valid_mask & finite_mask

        return DetectionSample(
            image = sample.image,
            boxes = boxes[valid_mask],
            labels= labels[valid_mask],
            image_id=sample.image_id,
            path=sample.path,
            orig_size=sample.orig_size,
        )

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")

        sample = self._load_sample(index)
        sample = self._clean_boxes(sample)

        if self._validate:
            sample.validate()

        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def generator(self):

        for index in range(len(self)):
            sample = self[index]
            yield sample.to_dict()

    def get_stats(self):
        total_boxes = 0
        class_counts = {name: 0 for name in self.class_names}
        
        for sample in self:
            total_boxes = total_boxes + len(sample.boxes)
            for label in sample.labels:
                class_name = class_name = self.index_to_class[label]
                class_counts[class_name] = class_counts + 1

        return {
            "num_samples": len(self),
            "total_boxes": total_boxes,
            "avg_boxes_per_image": total_boxes / len(self) if len(self) > 0 else 0,
            "class_distribution": class_counts,
        }
        