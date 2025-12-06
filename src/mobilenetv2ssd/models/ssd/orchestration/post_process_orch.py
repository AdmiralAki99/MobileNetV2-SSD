import tensorflow as tf
from typing import Any
from pathlib import Path

from mobilenetv2ssd.models.ssd.ops.postprocess_tf import decode_and_nms

def _read_deploy_config(config: dict[str, Any]):
    deploy_config = config['deploy']
    input_config = deploy_config['input']
    prior_config = deploy_config['priors']
    post_config = deploy_config['post_processing']
    classes_config = deploy_config['classes']
    deploy_config = {
        'input_size': {
            'image_height' : input_config.get('size',[224,224,3])[0],
            'image_width' : input_config.get('size',[224,224,3])[1],
        },
        'iou_thresh': float(post_config.get('nms_iou_threshold',0.4)),
        'variances': tf.constant(prior_config.get('variances',[0.1,0.2])),
        'max_detection': post_config.get('max_detections',100),
        'per_class_top_k': post_config.get('per_class_top_k',100),
        'score_thresh': post_config.get('score_threshold',0.45),
        'num_classes': classes_config.get('num_classes',21),
        'labels_map' : classes_config.get('labels_map',None),
        'use_sigmoid': classes_config.get('use_sigmoid',False)
    }
    return deploy_config

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

def _decode_class_names(class_id_tensor: tf.Tensor, labels: list[str]):
    class_ids = class_id_tensor.numpy()
    decoded = [[labels[i] for i in row] for row in class_ids]
    return decoded

def build_decoded_boxes(config: dict[str,Any], predicted_offsets: tf.Tensor, predicted_logits: tf.Tensor, priors: tf.Tensor):
    deploy_config = _read_deploy_config(config)

    # Decoding boxes
    nmsed_boxes,nmsed_scores, nmsed_classes, valid_detections = decode_and_nms(predicted_offsets = predicted_offsets, predicted_logits = predicted_logits, priors = priors, variances = deploy_config['variances'],scores_thresh = deploy_config['score_thresh'], iou_thresh = deploy_config['iou_thresh'], top_k = deploy_config['per_class_top_k'], max_detections = deploy_config['max_detection'],image_meta = deploy_config['input_size'],use_sigmoid = deploy_config['use_sigmoid'])   

    # Getting the classes
    classes = _load_label_map(Path('./voc_labels.txt'),use_sigmoid = deploy_config['use_sigmoid'])

    decoded_classes = _decode_class_names(nmsed_classes, classes)

    return nmsed_boxes, nmsed_scores, nmsed_classes, decoded_classes