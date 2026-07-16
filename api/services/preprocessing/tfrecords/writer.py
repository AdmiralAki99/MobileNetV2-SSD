import os
import boto3
from pathlib import Path
import tensorflow as tf

class TFRecordWriter:
    def __init__(self, output_dir, shard_size):
        self.output_dir= output_dir
        self.shard_size= shard_size
        self._is_s3= str(output_dir).startswith("s3://")
        self._shard_index = 0
        self._writer = None
        self._records_in_shard = 0
        
    def open(self, split_name):
        self.split_name = split_name
        shard_name= f"{self._shard_index:03d}.tfrecord"
        if self._is_s3:
            local_dir = Path("/tmp/tfrecord_shards")
            local_dir.mkdir(parents=True, exist_ok=True)
            self._local_path = local_dir / shard_name
            bucket, prefix = self.output_dir[5:].split("/",1)
            self._s3_bucket = bucket
            self._s3_key = f"{prefix}/{split_name}/{shard_name}"
        else:
            local_dir = Path(self.output_dir) / split_name
            local_dir.mkdir(parents= True, exist_ok= True)
            self._local_path = local_dir / shard_name
            
        self._writer = tf.io.TFRecordWriter(str(self._local_path))
        self._records_in_shard = 0
        
    def close(self):
        self._writer.close()
        if self._is_s3:
            boto3.client("s3").upload_file(str(self._local_path), self._s3_bucket, self._s3_key)
            os.remove(self._local_path)
            
    def write(self, sample):
        if self._records_in_shard >= self.shard_size:
            self.close()
            self._shard_index = self._shard_index + 1
            self.open(self.split_name)
            
        H, W = sample.orig_size
        with open(sample.path, "rb") as file:
            jpeg_bytes = file.read()
            
        feature = {
            "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpeg_bytes])),
            "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[H])),
            "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[W])),
            "image/boxes": tf.train.Feature(float_list=tf.train.FloatList(value=sample.boxes.flatten().tolist())),
            "image/boxes_count": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(sample.boxes)])),
            "image/labels": tf.train.Feature(int64_list=tf.train.Int64List(value=sample.labels.flatten().tolist())),
            "image/image_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample.image_id.encode("utf-8")])),
            "image/path": tf.train.Feature(bytes_list=tf.train.BytesList(value=[sample.path.encode("utf-8")])),
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self._writer.write(example.SerializeToString())
        self._records_in_shard = self._records_in_shard + 1
        