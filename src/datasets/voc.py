import numpy as np
from PIL import Image
from typing import Any
from pathlib import Path
import xml.etree.ElementTree as ET

from datasets.base import BaseDetectionDataset, DetectionSample

class VOCDataset(BaseDetectionDataset):
    def __init__(self, root: str | Path, split: str, classes_file: str | Path, use_difficult: bool = False):
        super().__init__(root, split, classes_file, use_difficult)

        self.jpeg_dir = self.root / "JPEGImages"
        self.annotation_dir = self.root / "Annotations"
        self.split_dir = self.root / "ImageSets" / "Main"

        # Validating the directories
        self._validate_paths()

        self.image_ids = self._load_image_ids()

        if len(self.image_ids) == 0:
            raise ValueError(f"No images found for split '{split}'")
        
    def _validate_paths(self):
        # Checking if the directory exists or not
        if not self.jpeg_dir.exists():
            raise FileNotFoundError(f"JPEGImages directory not found: {self.jpeg_dir}")
        
        if not self.annotation_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotation_dir}")

        if not self.split_dir.exists():
            raise FileNotFoundError(f"ImageSets/Main directory not found: {self.split_dir}")

    def _load_image_ids(self):
        # Loading the ids from the split file to get the proper images
        
        split_file = self.split_dir / f"{self.split}.txt"
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as file:
            ids = []
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split()
                    ids.append(parts[0])

        return ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Read the file
        image = Image.open(path).convert("RGB")
        return np.array(image, dtype = np.float32)

    def _parse_annotation(self, path: Path):

        if not path.exists():
             return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        # Loading the XML annotation
        tree = ET.parse(path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):

            # Getting the name
            name = (obj.findtext("name") or "").strip()
            if not name or name not in self.class_to_index:
                continue

            # Getting the difficult flag
            difficult = int(obj.findtext("difficult") or "0")
            if difficult and not self.use_difficult:
                continue

            # Getting the bounding box
            bbox = obj.find("bndbox")
            if bbox is None:
                continue

            try:
                x1 = float(bbox.findtext("xmin") or 0)
                y1 = float(bbox.findtext("ymin") or 0)
                x2 = float(bbox.findtext("xmax") or 0)
                y2 = float(bbox.findtext("ymax") or 0)
            except (ValueError, TypeError):
                continue

            # Making sure invalid boxes dont make it through
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_index[name])

        if boxes:
            return np.array(boxes, dtype= np.float32), np.array(labels, dtype=np.int32)
        else:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    def _load_sample(self, index: int):
        
        image_id = self.image_ids[index]

        # Loading the image
        image_path = self.jpeg_dir / f"{image_id}.jpg"
        image = self._load_image(image_path)

        # Loading the annotation
        annotation_path = self.annotation_dir / f"{image_id}.xml"
        boxes, labels = self._parse_annotation(annotation_path)

        return DetectionSample(
            image = image, 
            boxes = boxes,
            labels = labels,
            image_id = image_id,
            path = str(image_path),
            orig_size = image.shape[:2]
        )