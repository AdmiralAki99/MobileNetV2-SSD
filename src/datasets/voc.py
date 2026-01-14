import tensorflow as tf
from pathlib import Path
from typing import Any
import hashlib
import json
import xml.etree.ElementTree as ET

from datasets.base import BaseDetectionDataset

class VOCDataset(BaseDetectionDataset):
    def __init__(self, config: dict[str,Any], split: str, transform = None):
        super().__init__(config, split, transform)

        self._root = Path(config['data']['root'])
        self._train_split = config['data']['train_split']
        self._val_split = config['data']['val_split']
        self._use_difficult = bool(config["data"].get("use_difficult", False))

        # Creating the file directories
        self._jpeg_dir = self._root / "JPEGImages"
        self._annotation_dir = self._root / "Annotations"
        self._split_dir = self._root / "ImageSets" / "Main"

        if split in ("train", "trainval", "train_val"):
            split_name = self._train_split
        elif split in ("val", "validation"):
            split_name = self._val_split
        else:
            raise ValueError("Wrong Split Name Given")

        # Handling Split directories
        self._split_file = self._split_dir / f"{split_name}.txt"

        self._ids = self.read_ids(self._split_file)
        
        self._transform = transform
        
    def __len__(self):
        return len(self._ids)

    def _create_hash_signature(self, attributes: dict[str,Any]):
        serialized = json.dumps(attributes, sort_keys=True).encode()
        return hashlib.md5(serialized).hexdigest()

    def _load_raw_sample(self, index: int):
        # Load and decode the Image by reading the file, the annotations from the XML and map class names to the label
        image_id = self.get_image_id(index)
        
        jpeg_path = str(self._jpeg_dir / f"{image_id}.jpg")
        xml_path = str(self._annotation_dir / f"{image_id}.xml")
        
        image = tf.keras.utils.load_img(jpeg_path, color_mode="rgb")
        # Keeping the image in its raw format and will preprocess that later
        image = tf.keras.utils.img_to_array(image)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)

        boxes = []
        labels = []
        difficults = []

        # Reading the XML annotations

        tree = ET.parse(str(xml_path))
        
        root = tree.getroot()

        size = root.find('size')
        if size is None:
            height, width = int(image.shape[0]), int(image.shape[1])
        else:
            width = int(size.findtext("width") or 0)
            height = int(size.findtext("height") or 0)

        if width == 0 or height == 0:
            raise ValueError(f"Unknown width in {xml_path}")

        for annotation_obj in root.findall('object'):
            name = (annotation_obj.findtext("name", "") or "").strip()
            if not name:
                continue

            if name not in self._name_to_id:
                raise ValueError(f"Unknown class '{name}' in {xml_path}")

            difficult = int(annotation_obj.findtext("difficult","0") or "0")
            if (not self._use_difficult) and difficult == 1:
                continue

            bbox = annotation_obj.find("bndbox")
            if bbox is None:
                continue

            # Getting the Coordinates
            x1 = float(bbox.findtext("xmin", "nan"))
            y1 = float(bbox.findtext("ymin", "nan"))
            x2 = float(bbox.findtext("xmax", "nan"))
            y2 = float(bbox.findtext("ymax", "nan"))

            boxes.append([x1,y1,x2,y2])
            labels.append(int(self._name_to_id[name]))
            difficults.append(difficult)

        hash_signature_attributes = {
            'boxes' : boxes,
            'labels' : labels,
            'path': jpeg_path,
            'image_id': image_id,
            'width': width,
            'height': height
        }

        hash_signature = self._create_hash_signature(hash_signature_attributes)

        target = {
            'boxes' : boxes,
            'labels' : labels,
            'path': jpeg_path,
            'image_id': image_id,
            'hash_signature': hash_signature,
            'orig_size': tf.constant([width, height], dtype= tf.int32)
        }

        return image, target
        
    def get_image_id(self, index: int):
        if index < 0 or index >= len(self._ids):
            raise IndexError("Index length is out of bounds")
        return self._ids[index]

    def read_ids(self, file_path: str | Path):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        with open(file_path, "r") as f:
            labels = [line.strip().split(" ")[0] for line in f.readlines() if line.strip()]

        return labels
        
        
        
def build_voc_dataset(config: dict[str, Any], split: str, transform: None = None):

    dataset = VOCDataset(config, split = split, transform = transform)

    return dataset
    