import tensorflow as tf
from typing import Any
from pathlib import Path
from abc import ABC, abstractmethod

class BaseDetectionDataset(ABC):
    def __init__(self, config: dict[str,Any], split: str, transform = None):
        self._dataset_config = self.extract_dataset_config(config)
        self._labels = self.load_classes(self._dataset_config.get('classes_file', ''))
        self._id_to_name = {index + 1: element for index, element in enumerate(self._labels)}
        self._id_to_name[0] = "background"
        self._name_to_id = {element: index + 1 for index, element in enumerate(self._labels)}
        self._name_to_id["background"] = 0
        self._num_classes = len(self._labels) + 1

        self._split = split
        self._is_train = (split in ("train", "trainval"))
        self._transform = transform

    def extract_dataset_config(self, config: dict[str, Any]):
        data_config = config['data']
        augment_config = data_config.get('augment',{})
        normalization_config = data_config.get('normalization',{})
        
        tranform_config = {
            'random_flip': augment_config.get('random_flip', False),
            'random_flip_prob': augment_config.get('random_flip_prob', 0.5),
            'random_crop': augment_config.get('random_crop', False),
            'min_crop_iou_choices': augment_config.get('min_crop_iou_choices', []),
            'min_crop_scale': augment_config.get('min_crop_scale', 0.3),
            'max_crop_scale': augment_config.get('max_crop_scale', 1.0),
            'photometric_distort': augment_config.get('photometric_distort', False),
            'photometric_distort_prob': augment_config.get('photometric_distort_prob', 0.5),
            'normalization_mean': normalization_config.get('mean', [0.5,0.5,0.5]),
            'normalization_std': normalization_config.get('std', [0.25, 0.25, 0.25]),
        }

        dataset_config = {
            'root': data_config.get('root', ''),
            'input_size': tuple(data_config.get('input_size', [300,300])),
            'transform_opts': tranform_config,
            'classes_file': data_config.get('classes_file', '')
        }

        return dataset_config

    def load_classes(self, label_file_path: str | Path):
        if isinstance(label_file_path, str):
            label_file_path = Path(label_file_path)

        # Now reading the file and then loading it in
        with open(label_file_path, "r") as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]

        return labels

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _load_raw_sample(self, index: int):
        raise NotImplementedError

    def _standardize_target(self, image: tf.Tensor, target : dict, index: int):
        # Checking for keys in the targets
        validation_keys = ['boxes','labels']
        result = all(key in target for key in validation_keys)

        if not result:
            raise KeyError("target must contain 'boxes' and 'labels'")

        # Checking if the coordinates are in the xyxy format
        boxes = target['boxes']
        labels = target['labels']

        boxes = tf.convert_to_tensor(boxes)
        labels = tf.convert_to_tensor(labels)

        if boxes.shape.rank == 1:
            n = tf.shape(boxes)[0]

            boxes = tf.cond(tf.equal(n,0), lambda: tf.reshape(boxes, [0,4]), lambda: tf.reshape(boxes, [1,4]))

        if labels.shape.rank == 0:
            labels = tf.reshape(labels,[1])      

        # Checking and enforcing dtypes
        boxes = tf.cast(boxes,tf.float32)
        labels = tf.cast(labels, tf.int32)

        target['boxes'] = boxes
        target['labels'] = labels

        # Making sure the metadata exists
        if 'image_id' not in target:
            target['image_id'] = tf.constant(f'{index}',dtype= tf.string)
        else:
            # Convert to tensor with the initial value
            target['image_id'] = tf.constant(target['image_id'],dtype= tf.string)

        if 'hash_signature' not in target:
            target['hash_signature'] = tf.constant('',dtype= tf.string)
        else:
            target['hash_signature'] = tf.constant(target['hash_signature'],dtype= tf.string)

        if 'orig_size' not in target:
            target['orig_size'] = tf.shape(image)[0:2]

        if 'path' not in target:
            target['path'] = tf.constant("", dtype= tf.string)
        else:
            target['path'] = tf.constant(target['path'], dtype= tf.string)

        return target

    def _sanitize_target(self, image: tf.Tensor, target : dict):
        # Check for degenerate boxes which are x2 <= x1 or y2 <= y1
        boxes = target['boxes']
        labels = target['labels']

        finite_mask = tf.reduce_all(tf.math.is_finite(boxes), axis=-1)
        boxes = tf.boolean_mask(boxes, finite_mask)
        labels = tf.boolean_mask(labels, finite_mask)

        # Clip boxes to the original dimensions
        H,W = target['orig_size']

        H = tf.cast(H, tf.float32)
        W = tf.cast(W, tf.float32)
    
        x1, y1, x2, y2 = tf.split(boxes,num_or_size_splits = 4, axis = -1)

        x1 = tf.cast(x1, tf.float32)
        y1 = tf.cast(y1, tf.float32)
        x2 = tf.cast(x2, tf.float32)
        y2 = tf.cast(y2, tf.float32)

        x1 = tf.clip_by_value(x1, 0, W)
        y1 = tf.clip_by_value(y1, 0, H)
        x2 = tf.clip_by_value(x2, 0, W)
        y2 = tf.clip_by_value(y2, 0, H)

        boxes = tf.concat([x1, y1, x2, y2], axis = -1)

        x1, y1, x2, y2 = tf.split(boxes, num_or_size_splits = 4, axis = -1)

        degenerate_validity = tf.math.logical_or(x2 <= x1, y2 <= y1)
        degenerate_validity = tf.reshape(tf.math.logical_not(degenerate_validity),[-1])

        boxes = tf.boolean_mask(boxes,degenerate_validity)

        # Filtering the labels too
        labels = tf.boolean_mask(labels, degenerate_validity)

        target['boxes'] = boxes
        target['labels'] = labels

        return target
    

    def _validate_target(self, image: tf.Tensor, target : dict):
        # Validating the shape of the targets

        boxes = target['boxes']
        labels = target['labels']

        # Checking for Target boxes
        tf.debugging.assert_equal(tf.rank(boxes), 2, "boxes must be rank-2: [N,4]")
        tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, "boxes last dim must be 4")
        tf.debugging.assert_equal(tf.rank(labels), 1, "labels must be rank-1: [N]")
        tf.debugging.assert_equal(tf.shape(boxes)[0], tf.shape(labels)[0], "boxes and labels must have same N")
        

        tf.debugging.assert_equal(tf.reduce_all(tf.math.is_finite(boxes)), True, "boxes contain NaN/Inf")

        x1, y1, x2, y2 = tf.split(boxes,num_or_size_splits = 4, axis = -1)

        x1 = tf.cast(x1, tf.float32)
        y1 = tf.cast(y1, tf.float32)
        x2 = tf.cast(x2, tf.float32)
        y2 = tf.cast(y2, tf.float32)

        tf.debugging.assert_less_equal(x1, x2, message= " x1 <= x2 condition violated")
        tf.debugging.assert_less_equal(y1, y2, message= " y1 <= y2 condition violated")

        # Checking if the coordinates are in the bounds of the image
        image_shape = tf.shape(image)
    
        H = tf.cast(image_shape[0], tf.float32)
        W = tf.cast(image_shape[1], tf.float32)
    
        # Checking if the coordinates are within bounds
        x1_condition = tf.math.logical_and((x1 >= 0),(x1 <= W))
        x2_condition = tf.math.logical_and((x2 >= 0),(x2 <= W))
    
        x_validity = tf.reduce_all(tf.math.logical_and(x1_condition, x2_condition ))

        y1_condition = tf.math.logical_and((y1 >= 0),(y1 <= H))
        y2_condition = tf.math.logical_and((y2 >= 0),(y2 <= H))
    
        y_validity = tf.reduce_all(tf.math.logical_and(y1_condition, y2_condition))
    
        tf.debugging.assert_equal(x_validity, tf.constant(True, tf.bool), message = "Failed to validate x conditions")
        tf.debugging.assert_equal(y_validity, tf.constant(True, tf.bool), message = "Failed to validate y conditions")

        tf.debugging.assert_type(labels, tf.int32)
        tf.debugging.assert_greater_equal(tf.reduce_min(labels), 1, "labels must be >= 1 (0 is background)")

        # Checking for the other targets attributes
        tf.debugging.assert_equal(tf.rank(target["image_id"]), 0, "image_id must be scalar")
        tf.debugging.assert_equal(tf.rank(target["orig_size"]), 1, "orig_size must be rank-1")
        tf.debugging.assert_equal(tf.shape(target["orig_size"])[0], 2, "orig_size must be [2] (H,W)")
        tf.debugging.assert_less_equal(tf.reduce_max(labels),self._num_classes - 1,"label id out of range")

        # Checking the intensity values of an image
        tf.debugging.assert_equal(tf.rank(image), 3, "image must be [H,W,3]")
        tf.debugging.assert_equal(tf.shape(image)[-1], 3, "image must have 3 channels")

        tf.debugging.assert_equal(tf.rank(image),tf.constant(3, dtype = tf.int32), message = "The rank is not the same for the images")
        tf.debugging.assert_equal(tf.shape(image)[-1],tf.constant(3, dtype = tf.int32), message = "The channel dimension is invalid for image")
        tf.debugging.assert_equal(tf.math.reduce_all(tf.math.is_finite(tf.cast(image, dtype=tf.float32))), tf.constant(True, dtype= tf.bool), message = "The image intensities are not finite")
            
        tf.debugging.assert_equal(target['path'].dtype, tf.string, "path is invalid for targets") 
    
    @abstractmethod
    def get_image_id(self, index: int):
        raise NotImplementedError
    
    def __getitem__(self, index: int):
        image, target = self._load_raw_sample(index)

        target = self._standardize_target(image, target, index)

        target = self._sanitize_target(image, target)

        self._validate_target(image, target)

        if self._transform is not None:
            image, target = self._transform(image, target)
            self._validate_target(image, target)

        return image, target