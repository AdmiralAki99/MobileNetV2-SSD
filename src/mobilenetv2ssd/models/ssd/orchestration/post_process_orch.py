import tensorflow as tf
from typing import Any
from pathlib import Path

from mobilenetv2ssd.models.ssd.ops.postprocess_tf import decode_and_nms
from mobilenetv2ssd.core.precision_config import PrecisionConfig

def _read_eval_config(config: dict[str, Any]):
    eval_opts = config['eval']
    nms_config = eval_opts['nms']
    decode_config = eval_opts['decode']
    eval_config = {
        
        'iou_threshold': nms_config.get('iou_threshold', 0.5),
        'score_threshold': nms_config.get('score_threshold', 0.05),
        'max_detections_per_class': nms_config.get('max_detections_per_class', 50),
        'max_detections_per_image': nms_config.get('max_detections_per_image', 100),
        'per_class_top_k': nms_config.get('per_class_top_k', 100),
        'class_file': decode_config.get('root', None),
        'variances': tf.constant(list(decode_config.get('variances', [0.1,0.2])), dtype = tf.float32),
        'use_sigmoid': decode_config.get('use_sigmoid', False),
        'input_size': {'image_height': eval_opts.get('input', 0)[0],'image_width': eval_opts.get('input', 0)[1]}
    }
    
    return eval_config

def _load_label_map(label_file_path: str, use_sigmoid: bool = False):
    
    label_file_path = Path(label_file_path)
    
    with open(label_file_path, "r") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

    # Build lookup
    if use_sigmoid:
        label_dict = {idx: name for idx, name in enumerate(labels)}
    else:
        # Softmax so labels begin at 1
        label_dict = {idx + 1: name for idx, name in enumerate(labels)}
        label_dict[0] = "background"
        
    return label_dict

def _decode_class_names(class_id_tensor: tf.Tensor, labels: dict[int,str]):
    
    labels_list = tf.constant([labels[key] for key in sorted(labels)], dtype=tf.string)
    decoded_classes = tf.gather(labels_list,class_id_tensor)
    
    return decoded_classes

def build_decoded_boxes(config: dict[str,Any], predicted_offsets: tf.Tensor, predicted_logits: tf.Tensor, priors: tf.Tensor, precision_config: PrecisionConfig | None = None):
    eval_config = _read_eval_config(config)

    # Decoding boxes
    nmsed_boxes,nmsed_scores, nmsed_classes, valid_detections = decode_and_nms(predicted_offsets = predicted_offsets, predicted_logits = predicted_logits, priors = priors, variances = eval_config['variances'],scores_thresh = eval_config['score_threshold'], iou_thresh = eval_config['iou_threshold'], top_k = eval_config['per_class_top_k'], max_detections = eval_config['max_detections_per_image'],image_meta = eval_config['input_size'],use_sigmoid = eval_config['use_sigmoid'], precision_config= precision_config)   

    # Getting the classes
    classes = _load_label_map(eval_config['class_file'],use_sigmoid = eval_config['use_sigmoid'])

    decoded_classes = _decode_class_names(nmsed_classes, classes)

    return nmsed_boxes, nmsed_scores, nmsed_classes, decoded_classes, classes, valid_detections