import io
import boto3
from pathlib import Path
from typing import Any
import numpy as np
import tensorflow as tf
from PIL import Image
from .consensus import ConsensusAnnotation


class TFRecordWriter:
    def __init__(self, config: dict[str, Any]):
        self.output_dir = config.get("output", {}).get("tfrecords_dir", "datasets/etl_output/shards")
        self.shard_size = config.get("output", {}).get("shard_size", 1000)
        self._writer = None
        self._shard_index = 0
        self._records_in_shard = 0
        self._is_s3 = self.output_dir.startswith("s3://")  # Writer needs the file path

    def open(self, video_id: str, dataset_name: str = "unsorted"):
        self.video_id = video_id
        self.dataset_name = dataset_name
        shard_name = f"shard_{video_id}_{self._shard_index:05d}.tfrecord"
        if self._is_s3:
            local_dir = "/tmp/etl_shards"
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            self._local_path = f"{local_dir}/{shard_name}"
            bucket, prefix = self.output_dir[5:].split("/", 1)
            self._s3_bucket = bucket
            self._s3_key = f"{prefix}/{dataset_name}/{shard_name}"
        else:
            out = f"{self.output_dir}/{dataset_name}"
            Path(out).mkdir(parents=True, exist_ok=True)
            self._local_path = f"{out}/{shard_name}"

        self._writer = tf.io.TFRecordWriter(self._local_path)
        self._records_in_shard = 0
        self._shard_index = self._shard_index + 1

    def close(self):
        self._writer.close()
        if self._is_s3:  # Writer cannot write to S3, needs to be uploaded using boto3
            boto3.client("s3").upload_file(self._local_path, self._s3_bucket, self._s3_key)

    def write(self, image: np.ndarray, annotations: list[ConsensusAnnotation], image_id: str, path: str):

        H, W = image.shape[:2]

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        boxes_list, labels = [], []

        for annotation in annotations:
            boxes_list.extend(annotation.box.tolist())
            labels.append(annotation.class_id)

        boxes_count = len(annotations)

        boxes_feature = tf.train.Feature(float_list=tf.train.FloatList(value=boxes_list))
        boxes_count_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[boxes_count]))
        labels_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        height_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[H]))
        width_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[W]))
        image_id_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id.encode("utf-8")]))
        path_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[path.encode("utf-8")]))

        feature = {
            "image/encoded": image_feature,
            "image/height": height_feature,
            "image/width": width_feature,
            "image/boxes": boxes_feature,
            "image/boxes_count": boxes_count_feature,
            "image/labels": labels_feature,
            "image/image_id": image_id_feature,
            "image/path": path_feature,
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        self._writer.write(example.SerializeToString())

        self._records_in_shard = self._records_in_shard + 1

        if self._records_in_shard >= self.shard_size:
            self.close()
            self.open(self.video_id, self.dataset_name)
