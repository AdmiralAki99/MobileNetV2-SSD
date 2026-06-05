import numpy as np
from PIL import Image
from pathlib import Path

from datasets.base import BaseDetectionDataset, DetectionSample


class VisDroneDataset(BaseDetectionDataset):
    def __init__(self, root: str | Path, split: str, classes_file: str | Path, use_difficult: bool = False):
        super().__init__(root, split, classes_file, use_difficult=use_difficult)

        # The directory is in separate folders for images and annotations
        sub = f"VisDrone2019-DET-{self.split}"

        # Creating the data and annotation paths

        self._data_path = Path(self.root) / sub
        self._image_path = self._data_path / "images"
        self._annotation_path = self._data_path / "annotations"

        self._image_ids = self._load_image_ids()

        if len(self._image_ids) == 0:
            raise ValueError(f"No images found for split '{split}'")

    def _load_image_ids(self):
        # The image ids are the names of thej files
        image_ids = []

        # Iterating through the image directory to get the ids
        for image_file in self._image_path.glob("*.jpg"):
            image_ids.append(image_file.stem)

        return sorted(image_ids)

    def __len__(self):
        return len(self._image_ids)

    def _parse_annotation(self, annotation_path: Path):

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        # The annotation file is a text file with the format:
        # bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncated, occlusion

        boxes = []
        labels = []
        with open(annotation_path, "r") as file:
            for line in file:
                line = line.strip()
                # Checking if the line is not empty
                if line:
                    annotation_parts = line.split(",")
                    # Checking if the annotation has the correct number of parts
                    if len(annotation_parts) != 8:
                        continue

                    # Everything is correct till this point and now can be separated into its respective parts
                    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncated, occlusion = (
                        annotation_parts
                    )

                    # Checking some of the conditions for the annotation to be valid
                    if int(category) == 0 or int(score) == 0 or int(category) == 11:
                        continue

                    # These boxes are needed
                    x_min = int(bbox_left)
                    y_min = int(bbox_top)
                    x_max = int(bbox_left) + int(bbox_width)
                    y_max = int(bbox_top) + int(bbox_height)
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(category))

        return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int32)

    def _load_image(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path).convert("RGB")
        return np.array(image, dtype=np.float32)

    def iter_annotations(self):
        # This is for the K-Means clustering to get the anchor boxes
        for image_id in self._image_ids:
            annotation_path = self._annotation_path / f"{image_id}.txt"
            image_path = self._image_path / f"{image_id}.jpg"
            with Image.open(image_path) as img:
                width, height = img.size

            boxes, _ = self._parse_annotation(annotation_path)

            yield boxes, (height, width)

    def _load_sample(self, index):
        image_id = self._image_ids[index]

        # Loading the image
        image_path = self._image_path / f"{image_id}.jpg"
        image = self._load_image(image_path)

        # Loading the annotation
        annotation_path = self._annotation_path / f"{image_id}.txt"
        boxes, labels = self._parse_annotation(annotation_path)

        return DetectionSample(
            image=image,
            boxes=boxes,
            labels=labels,
            image_id=image_id,
            path=str(image_path),
            orig_size=(int(image.shape[0]), int(image.shape[1])),
        )
