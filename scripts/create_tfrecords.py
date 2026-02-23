import tensorflow as tf
from pathlib import Path
from typing import Any

from datasets.base import BaseDetectionDataset, DetectionSample, create_dataset_from_config

def _bytes_features(value):
    # This function takes values from String and Byte types
    return tf.train.Feature(bytes_list= tf.train.BytesList(value=value))

def _float_features(value):
    # This function takes values from Float and Double Values
    return tf.train.Feature(float_list= tf.train.FloatList(value=value))

def _int64_features(value):
    # This function takes values from (bool, enum, int)
    return tf.train.Feature(int64_list= tf.train.Int64List(value=value))

def parse_boxes(boxes):
    # Parsing the boxes
    flattened_boxes = boxes.flatten().tolist()
    return _float_features(flattened_boxes), _int64_features([len(boxes)])

def parse_labels(labels):
    flattened_labels = labels.flatten().tolist()
    return _int64_features(flattened_labels)

def parse_image_size(height, width):
    if isinstance(height,int):
       height= [height]

    if isinstance(width, int):
        width= [width]

    return _int64_features(height), _int64_features(width)

def parse_image_id(image_id):
    if isinstance(image_id, str):
        image_id = [image_id.encode('utf-8')]
        
    return _bytes_features(image_id)

def parse_raw_image(jpeg_bytes):
    if isinstance(jpeg_bytes,bytes):
        jpeg_bytes = [jpeg_bytes]

    return _bytes_features(jpeg_bytes)

def parse_path(path):
    if isinstance(path, str):
        path = [path.encode('utf-8')]
        
    return _bytes_features(path)

def encode_features(boxes: tf.train.Feature, boxes_count: tf.train.Feature, labels: tf.train.Feature, height: tf.train.Feature, width: tf.train.Feature, image_id: tf.train.Feature, image_bytes: tf.train.Feature, image_path: tf.train.Feature):
    feature = {
        'image/encoded': image_bytes,
        'image/height': height,
        'image/width': width,
        'image/boxes': boxes,
        'image/boxes_count': boxes_count,
        'image/labels': labels,
        'image/image_id': image_id,
        'image/path': image_path
    }

    example = tf.train.Example(features = tf.train.Features(feature= feature))
    return example.SerializeToString()

def encode_sample(sample: DetectionSample):
    # Encode an image sample
    
    boxes = sample.boxes
    labels = sample.labels
    H,W = sample.orig_size
    id_ = sample.image_id

    # Reading the JPEG file
    with open(sample.path,"rb") as file:
        jpeg_bytes = file.read()

    encoded_bytes = parse_raw_image(jpeg_bytes)
    encoded_boxes, encoded_boxes_count = parse_boxes(boxes)
    encoded_labels = parse_labels(labels)
    encoded_height, encoded_width = parse_image_size(H,W)
    encoded_image_id = parse_image_id(id_)
    encoded_path= parse_path(sample.path)

    # Encoding the features into image
    example = encode_features(boxes= encoded_boxes,boxes_count= encoded_boxes_count, labels= encoded_labels, height= encoded_height, width= encoded_width,image_id= encoded_image_id, image_bytes= encoded_bytes, image_path= encoded_path) 

    return example

def write_serialized_record(example, writer):
    if writer is None:
        raise ValueError("tf.io.TFRecordWriter is None")

    # Writing the example
    writer.write(example)
    
def write_split(dataset, output_dir: Path, split_name, shard_size= 500):
    shard_index= 0
    shard_img_count= 0

    output_dir= Path(output_dir)
    main_dir= output_dir / split_name
    main_dir.mkdir(parents= True, exist_ok= True)
    
    existing = list(main_dir.glob("*.tfrecord"))
    if existing:
        raise FileExistsError(
            f"{main_dir} already contains {len(existing)} shard(s). "
            "Delete the directory or choose a different output_dir."
        )
    
    output_dir= output_dir / split_name / f"{shard_index:03d}.tfrecord"
    writer= tf.io.TFRecordWriter(str(output_dir))

    # Looping through the dataset
    for sample in dataset:
        # Creating the record
        record = encode_sample(sample)
        write_serialized_record(record, writer)
        shard_img_count= shard_img_count + 1

        if shard_img_count >= shard_size:
            # Creating a new shard
            writer.close()
            print(f"Finished Writing Shard: {shard_index:03d}")
            shard_index= shard_index + 1
            shard_img_count= 0
            shard_dir = main_dir / f"{shard_index:03d}.tfrecord"
            writer= tf.io.TFRecordWriter(str(shard_dir))

    writer.close()
    
def create_tfrecords_from_dataset(config: dict[str, Any], shard_size= 500):
    
    output_dir = Path(config['data']['root']) / "shards"
    
    training_dataset = create_dataset_from_config(config= config, split= config['data']['train_split'])
    print(f"Training Dataset of {training_dataset.__class__.__name__}, with split: { config['data']['train_split']}..{'.'*20}")
    write_split(dataset= training_dataset, output_dir= output_dir, split_name= training_dataset.split,shard_size= shard_size)
    print(f"Finished Training Dataset sharding with {len(training_dataset)//shard_size} shards..{'.'*20}")
    
    val_dataset = create_dataset_from_config(config= config, split= config['data']['val_split'])
    print(f"Validation Dataset of {val_dataset.__class__.__name__}, with split: { config['data']['val_split']}..{'.'*20}")
    write_split(dataset= val_dataset, output_dir= output_dir, split_name= val_dataset.split, shard_size= shard_size)
    print(f"Finished Validation Dataset sharding with {len(val_dataset)//shard_size} shards..{'.'*20}")
    
    
if __name__ == "__main__":
    from mobilenetv2ssd.core.config import load_config
    
    config = load_config("../configs/experiments/exp001_baseline.yaml")
    create_tfrecords_from_dataset(config= config)