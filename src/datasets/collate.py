import tensorflow as tf
from typing import Any
from pathlib import Path

from datasets.base import BaseDetectionDataset
from datasets.voc import VOCDataset
from datasets.transforms import Compose, build_train_transforms

_TFRECORD_FEATURE_DESCRIPTION = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/boxes': tf.io.VarLenFeature(tf.float32),
    'image/boxes_count': tf.io.FixedLenFeature([], tf.int64),
    'image/labels': tf.io.VarLenFeature(tf.int64),
    'image/image_id': tf.io.FixedLenFeature([], tf.string),
    'image/path': tf.io.FixedLenFeature([], tf.string)
}

_OUTPUT_SIGNATURE = {
    "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
    "boxes": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
    "labels": tf.TensorSpec(shape=(None,), dtype=tf.int32),
    "image_id": tf.TensorSpec(shape=(), dtype=tf.string),
    "path": tf.TensorSpec(shape=(), dtype=tf.string),
    "orig_size": tf.TensorSpec(shape=(2,), dtype=tf.int32),
}

def _parse_tfrecord(proto: tf.Tensor):
    # This function is mapped over a TFRecordDataset
    # Need to now match the info to the _OUTPUT_SIGNTURE
    
    example = tf.io.parse_single_example(serialized=proto, features= _TFRECORD_FEATURE_DESCRIPTION)
    # Image Operations
    image= tf.io.decode_jpeg(example['image/encoded'],channels= 3)
    image= tf.cast(image, dtype= tf.float32)
    # Box Operations
    boxes_count = example['image/boxes_count']
    boxes = tf.sparse.to_dense(example['image/boxes'])
    boxes = tf.reshape(boxes,[boxes_count,4])
    # Labels Operations
    labels= tf.cast(tf.sparse.to_dense(example['image/labels']),dtype= tf.int32)
    # Metadata Operations
    orig_size = tf.cast(tf.stack([example['image/height'],example['image/width']], axis=-1), dtype= tf.int32)
    path= example['image/path']
    image_id = example['image/image_id']
    
    return {
        "image": image,
        "boxes": boxes,
        "labels": labels,
        "image_id": image_id,
        "path": path,
        "orig_size": orig_size
    }
    
    
    
def create_training_dataset_config(config: dict[str, Any]):
    training_dataset_opts = config['data'].get('train', {})

    dataset_opts = {
        'batch_size': training_dataset_opts.get('batch_size', 5),
        'padded_shapes': {
            'boxes': training_dataset_opts.get('padded_shapes', {}).get('boxes', [None, 4]),
            'image': training_dataset_opts.get('padded_shapes', {}).get('image', [None,None, 3]),
            'labels': training_dataset_opts.get('padded_shapes', {}).get('labels', [None]),
            'image_id': training_dataset_opts.get('padded_shapes', {}).get('image_id', []),
            'path': training_dataset_opts.get('padded_shapes', {}).get('path', []),
            'orig_size': [2] # TODO: Add this to the config file in the future DIR: configs/data/*.yaml
        },
        'padding_values': {
            'image': training_dataset_opts.get('padding_values', {}).get('image', 0.0),
            'boxes': training_dataset_opts.get('padding_values', {}).get('boxes', -1.0),
            'labels': training_dataset_opts.get('padding_values', {}).get('labels', 0),
        },
        'shuffle': training_dataset_opts.get('shuffle', False),
        'prefetch': training_dataset_opts.get('prefetch', False),
        'max_boxes': training_dataset_opts.get('max_boxes', 100),
        'repeat': training_dataset_opts.get('repeat', False),
    }

    return dataset_opts

def create_validation_dataset_config(config: dict[str, Any]):
    validation_dataset_opts = config['data'].get('val', {})

    dataset_opts = {
        'batch_size': validation_dataset_opts.get('batch_size', 5),
        'padded_shapes': {
            'boxes': validation_dataset_opts.get('padded_shapes', {}).get('boxes', [None, 4]),
            'image': validation_dataset_opts.get('padded_shapes', {}).get('image', [None,None, 3]),
            'labels': validation_dataset_opts.get('padded_shapes', {}).get('labels', [None]),
            'image_id': validation_dataset_opts.get('padded_shapes', {}).get('image_id', []),
            'path': validation_dataset_opts.get('padded_shapes', {}).get('path', []),
            'orig_size': [2] ## TODO: Add this to the config file in the future DIR: configs/data/*.yaml
        },
        'padding_values': {
            'image': validation_dataset_opts.get('padding_values', {}).get('image', 0.0),
            'boxes': validation_dataset_opts.get('padding_values', {}).get('boxes', -1.0),
            'labels': validation_dataset_opts.get('padding_values', {}).get('labels', 0),
        },
        'shuffle': validation_dataset_opts.get('shuffle', False),
        'prefetch': validation_dataset_opts.get('prefetch', False),
        'max_boxes': validation_dataset_opts.get('max_boxes', 100),
    }

    return dataset_opts

def _create_gt_mask(data):
    gt_mask = data['labels'] > 0
    data['gt_mask'] = gt_mask
    return data

def apply_transform(sample, transform):
        image = sample["image"]
        target = {
            "boxes": sample["boxes"],
            "labels": sample["labels"],
            "orig_size": sample["orig_size"],
        }
        image, target = transform(image, target)
        return {**sample, "image": image, "boxes": target["boxes"], "labels": target["labels"]}
    
def create_validation_dataset(config: dict[str, Any], dataset: BaseDetectionDataset, transform: Compose):
    dataset_opts = create_validation_dataset_config(config)

    tf_dataset = tf.data.Dataset.from_generator(dataset.generator,output_signature = _OUTPUT_SIGNATURE)

    # Now checking for the options
    if dataset_opts['shuffle']:
        buffer_size = min(1000, len(dataset))
        tf_dataset = tf_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

    # Adding the transforms
    tf_dataset = tf_dataset.map(lambda x: apply_transform(x, transform), num_parallel_calls=tf.data.AUTOTUNE)

    # Padding the batch is crucial for the dataset
    tf_dataset = tf_dataset.padded_batch(batch_size = dataset_opts['batch_size'], padded_shapes = dataset_opts['padded_shapes'], padding_values = {
        'boxes' : tf.constant(dataset_opts['padding_values']['boxes'], tf.float32),
        'image' : tf.constant(dataset_opts['padding_values']['image'], tf.float32),
        'labels' : tf.constant(dataset_opts['padding_values']['labels'], tf.int32),
        'image_id' : tf.constant('', tf.string),
        'path' : tf.constant('', tf.string),
        'orig_size': tf.constant(0, tf.int32)
    })

    # Mapping a valid mask function
    tf_dataset = tf_dataset.map(_create_gt_mask, num_parallel_calls = tf.data.AUTOTUNE)
    


    # # Adding prefetch
    # if dataset_opts['prefetch']:
    #     tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset

def create_training_dataset(dataset: BaseDetectionDataset, config: dict[str,Any], transform: Compose):

    dataset_opts = create_training_dataset_config(config)

    tf_dataset = tf.data.Dataset.from_generator(dataset.generator, output_signature = _OUTPUT_SIGNATURE)

    # Shuffling the dataset
    if dataset_opts['shuffle']:
        buffer_size = min(1000, len(dataset))
        tf_dataset = tf_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

    # Turning on repeat
    if dataset_opts['repeat']:
        tf_dataset = tf_dataset.repeat()
        
    # Adding the transforms
    tf_dataset = tf_dataset.map(lambda x: apply_transform(x, transform), num_parallel_calls=tf.data.AUTOTUNE)

    tf_dataset = tf_dataset.padded_batch(batch_size = dataset_opts['batch_size'], padded_shapes = dataset_opts['padded_shapes'], padding_values = {
        'boxes' : tf.constant(-1, tf.float32),
        'image' : tf.constant(0, tf.float32),
        'labels' : tf.constant(0, tf.int32),
        'image_id' : tf.constant('', tf.string),
        'path' : tf.constant('', tf.string),
        'orig_size': tf.constant(0, tf.int32)
    })

    # Mapping a valid mask function
    tf_dataset = tf_dataset.map(_create_gt_mask, num_parallel_calls = tf.data.AUTOTUNE)

    # Adding prefetch
    if dataset_opts['prefetch']:
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset


def create_training_dataset_from_tfrecords(config: dict[str, Any], shard_paths: list[str] | None, transform: Compose):
    dataset_opts = create_training_dataset_config(config)
    
    if shard_paths is None:
        # Need to resolve it
        shard_dir = Path(config['data']['root']) / "shards" / config['data']['train_split']
        shard_paths = [str(path) for path in shard_dir.iterdir() if path.is_file()]
        
    tf_dataset = tf.data.TFRecordDataset(filenames= shard_paths, num_parallel_reads= tf.data.AUTOTUNE).map(_parse_tfrecord, num_parallel_calls= tf.data.AUTOTUNE)

    if dataset_opts['shuffle']:
        buffer_size = 2000 # TODO: Think about a config fallback
        tf_dataset = tf_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        
    if dataset_opts['repeat']:
        tf_dataset = tf_dataset.repeat()
        
    tf_dataset = tf_dataset.map(lambda x: apply_transform(x, transform), num_parallel_calls=tf.data.AUTOTUNE)
    
    tf_dataset = tf_dataset.padded_batch(batch_size = dataset_opts['batch_size'], padded_shapes = dataset_opts['padded_shapes'], padding_values = {
        'boxes' : tf.constant(-1, tf.float32),
        'image' : tf.constant(0, tf.float32),
        'labels' : tf.constant(0, tf.int32),
        'image_id' : tf.constant('', tf.string),
        'path' : tf.constant('', tf.string),
        'orig_size': tf.constant(0, tf.int32)
    })
    
    tf_dataset = tf_dataset.map(_create_gt_mask, num_parallel_calls = tf.data.AUTOTUNE)
    
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE) # High speed rails so there is no need to stop this if sharding is done
    
    return tf_dataset


def create_validation_dataset_from_tfrecords(config: dict[str, Any], shard_paths: list[str] | None, transform: Compose):
    dataset_opts = create_validation_dataset_config(config)

    if shard_paths is None:
        # Need to resolve it
        shard_dir = Path(config['data']['root']) / "shards" / config['data']['val_split']
        shard_paths = [str(path) for path in shard_dir.iterdir() if path.is_file()]
        
    tf_dataset = tf.data.TFRecordDataset(filenames= shard_paths, num_parallel_reads= tf.data.AUTOTUNE).map(_parse_tfrecord, num_parallel_calls= tf.data.AUTOTUNE)
    
    # Now checking for the options
    if dataset_opts['shuffle']:
        buffer_size = 2000
        tf_dataset = tf_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

    # Adding the transforms
    tf_dataset = tf_dataset.map(lambda x: apply_transform(x, transform), num_parallel_calls=tf.data.AUTOTUNE)

    # Padding the batch is crucial for the dataset
    tf_dataset = tf_dataset.padded_batch(batch_size = dataset_opts['batch_size'], padded_shapes = dataset_opts['padded_shapes'], padding_values = {
        'boxes' : tf.constant(dataset_opts['padding_values']['boxes'], tf.float32),
        'image' : tf.constant(dataset_opts['padding_values']['image'], tf.float32),
        'labels' : tf.constant(dataset_opts['padding_values']['labels'], tf.int32),
        'image_id' : tf.constant('', tf.string),
        'path' : tf.constant('', tf.string),
        'orig_size': tf.constant(0, tf.int32)
    })

    # Mapping a valid mask function
    tf_dataset = tf_dataset.map(_create_gt_mask, num_parallel_calls = tf.data.AUTOTUNE)

    # Adding prefetch
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset

        