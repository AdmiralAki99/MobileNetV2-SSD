import tensorflow as tf
from typing import Any

from mobilenetv2ssd.models.ssd.ops.match_ops_tf import match_priors
from mobilenetv2ssd.models.ssd.ops.encode_ops_tf import encode_boxes_core, encode_boxes_batch

def _extract_information_from_train_config(config : dict[str, Any]):
    model_config = config['model']
    train_config = config['train']
    target_config = {
        "variances": model_config['priors'].get("variances",[0.1,0.2]),
        'image_size': model_config.get("input_size",[224,224]),
        'iou_threshold_pos': model_config['matcher'].get("iou_threshold_pos",0.5),
        'iou_threshold_neg': model_config['matcher'].get("iou_threshold_neg",0.4),
        'allow_low_quality_matches': model_config['matcher'].get("allow_low_quality_matches",True), # Bipartite flag
        'center_in_gt': model_config['matcher'].get("center_in_gt",False),
        'neg_pos_ratio': train_config['sampler'].get('neg_pos_ratio',3.0),
        'min_neg': train_config['sampler'].get('min_neg',0),
        'max_neg': train_config['sampler'].get('max_neg',None),
        'diagnostics': True
    }

    return target_config

def building_training_targets(config: dict[str, Any], priors_cxcywh: tf.Tensor, gt_boxes_xyxy: tf.Tensor, gt_labels: tf.Tensor, gt_valid_mask : tf.Tensor):
    # This is the orchestrator for building the training targets
    # Steps:
    # 1. Extract the configuration used to create the matches from train & model config
    # 2. Sanitize GT Boxes (Need to check if needed if not offload to preprocessing stage)
    # 3. Match Priors to GT boxes for one image (batch using tf.map_fn)
    # 4. Encode to Offsets for the matched GT boxes and Priors(batch using tf.map_fn)
    # 5. Check if Diagnostics are needed (matched_iou, num_pos)
    target_config = _extract_information_from_train_config(config)
    
    batched_output = {
        "matched_gt_xyxy": tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        "matched_gt_labels": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "pos_mask": tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "neg_mask": tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "ignore_mask": tf.TensorSpec(shape=(None,), dtype=tf.bool),
        "matched_gt_idx": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "num_pos": tf.TensorSpec(shape=(), dtype=tf.int32),
        "matched_iou": tf.TensorSpec(shape=(None,), dtype=tf.float32),
    }

    matched_dict = tf.map_fn(lambda inputs: match_priors(priors_cxcywh = priors_cxcywh, gt_boxes_xyxy = inputs[0], gt_labels = inputs[1],gt_valid_mask = inputs[2],positive_iou_thresh = target_config['iou_threshold_pos'], negative_iou_thresh = target_config['iou_threshold_neg'],max_pos_per_gt = None,allow_low_qual_matches = target_config['allow_low_quality_matches'],center_in_gt = target_config['center_in_gt'],return_iou = target_config['diagnostics']), 
                             elems = (gt_boxes_xyxy, gt_labels, gt_valid_mask), 
                             fn_output_signature = batched_output) 
    
    localization_targets =  encode_boxes_batch(matched_gt_xyxy = matched_dict['matched_gt_xyxy'],priors_cxcywh = priors_cxcywh, variances = tuple(target_config['variances']))
    
    # Get the values from the dict
    
    classification_targets = matched_dict["matched_gt_labels"]
    pos_mask = matched_dict["pos_mask"]
    neg_mask = matched_dict["neg_mask"]
    ignore_mask = matched_dict["ignore_mask"]
    
    if target_config['diagnostics']:
        diagnostics = {'matched_iou': matched_dict["matched_iou"], 'num_pos': matched_dict["num_pos"]}
    else:
        diagnostics = None
        
    return localization_targets, classification_targets, pos_mask, neg_mask, ignore_mask, diagnostics