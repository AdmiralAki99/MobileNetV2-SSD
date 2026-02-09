import tensorflow as tf
from mobilenetv2ssd.core.logger import Logger
from mobilenetv2ssd.core.fingerprint import Fingerprint
from typing import Any
from pathlib import Path
import json

from mobilenetv2ssd.models.ssd.ops.box_ops_tf import iou_matrix_core

def ssd_get_prior_stats(positive_mask: tf.Tensor, negative_mask: tf.Tensor):
    positive_mask = tf.cast(positive_mask, tf.int32)
    negative_mask = tf.cast(negative_mask, tf.int32)

    # Per image values
    positive_prior_per_image = tf.reduce_sum(positive_mask, axis = 1)
    negative_prior_per_image = tf.reduce_sum(negative_mask, axis = 1)

    # Number of positive & negative anchors
    number_positive = tf.reduce_sum(positive_prior_per_image)
    number_negative = tf.reduce_sum(negative_prior_per_image)

    return{
        "num_pos": int(number_positive.numpy()),
        "pos_min": int(tf.reduce_min(positive_prior_per_image).numpy()),
        "pos_mean": float(tf.reduce_mean(tf.cast(positive_prior_per_image, tf.float32)).numpy()),
        "pos_max": int(tf.reduce_max(positive_prior_per_image).numpy()),
        "num_neg": int(number_negative.numpy()),
        "neg_pos_ratio": float((tf.cast(number_negative, tf.float32) / tf.maximum(tf.cast(number_positive, tf.float32), 1.0)).numpy()),
        "zero_pos_frac": float(tf.reduce_mean(tf.cast(positive_prior_per_image == 0, tf.float32)).numpy()), 
    }
    
def calculate_model_prediction_health(predicted_logits: tf.Tensor, predicted_offsets: tf.Tensor, logger: Logger):
    
    # Calculating the information on the predicted logits
    probs_correct = tf.nn.softmax(predicted_logits, axis=-1)  # correct (classes)
    probs_wrong = tf.nn.softmax(predicted_logits, axis=1)   # wrong (priors)
    probabilities = tf.nn.softmax(predicted_logits, axis=-1)
    
    background_probs = probabilities[..., 0]
    foreground_probs = probabilities[..., 1:]
    
    logger.metric(f"Sum over classes (Should be 1): {tf.reduce_mean(tf.reduce_sum(probs_correct, axis=-1)).numpy()}")
    logger.metric(f"Sum over classes (should be 1 only if axis=1 softmax): {tf.reduce_mean(tf.reduce_sum(probs_wrong, axis=1)).numpy()}")
    
    logger.metric(f"Mean background probability: {tf.reduce_mean(background_probs).numpy()}")
    logger.metric(f"Max background probability: {tf.reduce_max(background_probs).numpy()}")
    
    logger.metric(f"Mean top foreground probability: {tf.reduce_mean(tf.reduce_max(foreground_probs, axis=-1)).numpy()}")
    logger.metric(f"Mean sum foreground probability: {tf.reduce_mean(tf.reduce_sum(foreground_probs, axis=-1)).numpy()}")
    logger.metric(f"Max foreground probability: {tf.reduce_max(tf.reduce_sum(foreground_probs, axis=-1)).numpy()}")
    
    logger.metric(f"Predicted Logits mean: {tf.reduce_mean(predicted_logits).numpy()}")
    logger.metric(f"Predicted Logits std: {tf.math.reduce_std(predicted_logits).numpy()}")
    logger.metric(f"Predicted Logits max: {tf.reduce_max(predicted_logits).numpy()}")
    logger.metric(f"Predicted Logits min: {tf.reduce_min(predicted_logits).numpy()}")
    
def calculate_nms_health_scores(pred_scores: tf.Tensor, valid_detections: tf.Tensor):
    
    K = tf.shape(pred_scores)[1]
    indices = tf.range(K)[tf.newaxis,:]
    
    valid_mask = indices < valid_detections[:, tf.newaxis]
    
    valid_scores = tf.boolean_mask(pred_scores, valid_mask)
    
    # Num of valid total scores
    num_valid = tf.size(valid_scores)

    # Min valid score
    min_valid = tf.cond(num_valid > 0, lambda: tf.reduce_min(valid_scores), lambda: tf.constant(0.0, dtype= pred_scores.dtype))

    # Mean valid score
    mean_valid = tf.cond(num_valid > 0, lambda: tf.reduce_mean(valid_scores), lambda: tf.constant(0.0, dtype= pred_scores.dtype))

    # Max valid score
    max_valid = tf.cond(num_valid > 0, lambda: tf.reduce_max(valid_scores), lambda: tf.constant(0.0, dtype= pred_scores.dtype))
    
    below = tf.reduce_sum(tf.cast(valid_scores < 0.9, tf.int32))

    # Average valid detections
    average_valid_detections = tf.reduce_mean(tf.cast(valid_detections, tf.float32))

    # Zero detections fractions
    zero_detections_fraction = tf.reduce_mean(tf.cast(valid_detections == 0, tf.float32))

    neg_eps = tf.constant(1e-9, dtype = pred_scores.dtype)
    masked_scores = tf.where(valid_mask, pred_scores, neg_eps)

    top1_per_image = tf.reduce_max(masked_scores, axis = 1)

    valid_top1 = tf.boolean_mask(top1_per_image, valid_detections > 0)

    # Mean Top 1 scores
    mean_top1 = tf.cond(tf.size(valid_top1) > 0, lambda: tf.reduce_mean(valid_top1), lambda: tf.constant(0.0, dtype=pred_scores.dtype))

    # Top 1 including 0 detections
    top1_incl0 = tf.where(valid_detections > 0, top1_per_image, tf.zeros_like(top1_per_image))

    # Mean Top1 including 0 detections
    mean_top1_incl0 = tf.reduce_mean(top1_incl0)

    return {
        'num_valid': num_valid,
        'min_valid': min_valid,
        'mean_valid': mean_valid,
        'max_valid': max_valid,
        'below_thresh_scores': below,
        'average_valid_det': average_valid_detections,
        'zero_valid_det': zero_detections_fraction,
        'mean_top1': mean_top1,
        'top1_incl0': top1_incl0,
        'mean_top1_incl0': mean_top1_incl0
    }
    
def calculate_gt_health_scores(ground_truth_boxes: tf.Tensor, ground_truth_labels: tf.Tensor, ground_truth_mask: tf.Tensor, select_k:int = 10):
    
    ground_truth_count = tf.reduce_sum(tf.cast(ground_truth_mask, dtype= tf.int32), axis = -1)
    avg_ground_truth_boxes_per_image = tf.reduce_mean(ground_truth_count)

    zero_ground_truth_mask = ground_truth_count == 0
    zero_ground_truth_ratio = tf.reduce_mean(tf.cast(zero_ground_truth_mask, dtype= tf.int32))

    ground_truth_per_batch = tf.boolean_mask(ground_truth_labels, ground_truth_mask)

    if tf.size(ground_truth_per_batch) != 0:
        
        unique_classes, indices, counts = tf.unique_with_counts(ground_truth_per_batch)
    
        desc_order = tf.argsort(counts, direction= 'DESCENDING')
        unique_classes = tf.gather(unique_classes, desc_order)
        counts = tf.gather(counts, desc_order)

        select_k = tf.minimum(select_k, tf.size(unique_classes))
        select_classes = tf.gather(unique_classes, tf.range(select_k))
        select_counts = tf.gather(counts, tf.range(select_k))
    else:
        select_classes = tf.constant([], dtype= tf.int32)
        select_counts = tf.constant([], dtype= tf.int32)

    return {
        'ground_truth_count': ground_truth_count,
        'avg_ground_truth_boxes_per_image': avg_ground_truth_boxes_per_image,
        'zero_ground_truth_ratio': zero_ground_truth_ratio,
        'top_gt_classes': select_classes.numpy().tolist(),
        'top_gt_class_counts': select_counts.numpy().tolist(),
        'top_gt_class_distribution': tf.repeat(select_classes, repeats= select_counts)
    }
    
def calculate_pred_health_metrics(pred_scores: tf.Tensor, pred_labels: tf.Tensor, valid_detections: tf.Tensor, select_k: int = 10, background_id: int = 0):
    
    K = tf.shape(pred_scores)[1]
    indices = tf.range(K)[tf.newaxis,:]

    valid_mask = indices < valid_detections[:, tf.newaxis]

    classes_per_batch = tf.boolean_mask(pred_labels, valid_mask)
    scores_per_batch = tf.boolean_mask(pred_scores, valid_mask)

    # Getting the foreground classes
    foreground_mask = classes_per_batch != background_id
    classes_per_batch = tf.boolean_mask(classes_per_batch, foreground_mask)
    scores_per_batch = tf.boolean_mask(scores_per_batch, foreground_mask)

    # Checking if there is something to return
    if classes_per_batch.shape.rank == 0:
        return {
        'top_classes': [],
        'top_class_countes': [],
        'top_class_distribution': tf.constant([], dtype= tf.int32)
        }

    # Saved me so much time by this implementation
    unique_classes, indices, counts = tf.unique_with_counts(classes_per_batch)

    desc_order = tf.argsort(counts, direction= 'DESCENDING')
    unique_classes = tf.gather(unique_classes, desc_order)
    counts = tf.gather(counts, desc_order)

    # Now Taking only a subset of the values for easy logging
    select_k = tf.minimum(select_k, tf.size(unique_classes))
    select_classes = tf.gather(unique_classes, tf.range(select_k))
    select_counts = tf.gather(counts, tf.range(select_k))

    return {
        'top_classes': select_classes.numpy().tolist(),
        'top_class_counts': select_counts.numpy().tolist(),
        'top_class_distribution': tf.repeat(select_classes, repeats= select_counts)
    }

def verify_pred_boxes_sanity(pred_boxes: tf.Tensor, valid_detections):
    K = tf.shape(pred_boxes)[1]
    indices = tf.range(K)[tf.newaxis,:]

    valid_mask = indices < valid_detections[:, tf.newaxis]
    valid_boxes = tf.boolean_mask(pred_boxes, valid_mask)

    if tf.size(valid_boxes) == 0:
        return {
            'min_coordinates': [],
            'max_coordinates': []
        }

    valid_boxes = tf.reshape(valid_boxes, [-1,4])
    min_coordinate = tf.reduce_min(valid_boxes, axis= 0)
    max_coordinate = tf.reduce_max(valid_boxes, axis= 0)

    return {
        'min_coordinates': min_coordinate.numpy().tolist(),
        'max_coordinates': max_coordinate.numpy().tolist()
    }
    
def gt_box_range(ground_truth_boxes: tf.Tensor, ground_truth_mask: tf.Tensor):
    
    ground_truth_per_batch = tf.boolean_mask(ground_truth_boxes, ground_truth_mask)

    if tf.size(ground_truth_per_batch) == 0:
        return {
            'min_coordinates': [],
            'max_coordinates': []
        }

    valid_boxes = tf.reshape(ground_truth_per_batch, [-1,4])
    min_coordinate = tf.reduce_min(valid_boxes, axis= 0)
    max_coordinate = tf.reduce_max(valid_boxes, axis= 0)

    return {
        'min_coordinates': min_coordinate.numpy().tolist(),
        'max_coordinates': max_coordinate.numpy().tolist()
    }

def prediction_box_bad_frac(pred_boxes: tf.Tensor, valid_detections: tf.Tensor):
    B = pred_boxes.shape[0]
    K = pred_boxes.shape[1]

    valid_mask = tf.range(K)[tf.newaxis, :] < valid_detections[:, tf.newaxis]

    x_min, y_min, x_max, y_max = tf.split(pred_boxes, 4, axis=-1)
    bad_boxes = tf.squeeze(tf.logical_or(x_max <= x_min, y_max <= y_min), axis = -1)

    valid_bad_boxes = tf.boolean_mask(bad_boxes, valid_mask)
    if tf.size(valid_bad_boxes) == 0:
        return 0.0

    return float(tf.reduce_mean(tf.cast(valid_bad_boxes, tf.float32)).numpy())

def ground_truth_box_bad_frac(gt_boxes: tf.Tensor, gt_mask: tf.Tensor):
    
    x_min, y_min, x_max, y_max = tf.split(gt_boxes, 4, axis=-1)
    bad_boxes = tf.squeeze(tf.logical_or(x_max <= x_min, y_max <= y_min), axis = -1)
    valid_bad_boxes = tf.boolean_mask(bad_boxes, gt_mask)

    if tf.size(valid_bad_boxes) == 0:
        return 0.0

    return float(tf.reduce_mean(tf.cast(valid_bad_boxes, tf.float32)).numpy())

def calculate_iou_sanity_top1(pred_boxes: tf.Tensor, pred_scores: tf.Tensor, valid_detections: tf.Tensor, ground_truth_boxes: tf.Tensor, gt_mask: tf.Tensor, include_no_detection_as_zero: bool = True):
    B = tf.shape(pred_boxes)[0]
    K = tf.shape(pred_boxes)[1]

    indices = tf.range(K)[tf.newaxis,:]
    valid_mask = indices < valid_detections[:, tf.newaxis]

    # For Scores
    neg_eps = tf.constant(-1e9, dtype=pred_scores.dtype)
    masked_scores = tf.where(valid_mask, pred_scores, neg_eps)
    top_index = tf.argmax(masked_scores, axis= 1, output_type= tf.int32)

    top_boxes = tf.gather(pred_boxes, top_index, batch_dims= 0)

    ious_only_det = []
    iou_incl0 = []
    num_gt_images = 0
    num_det_images = 0
    
    for index in range(int(B.numpy())):
        gt_box_image = tf.boolean_mask(ground_truth_boxes[index], gt_mask[index])
        if tf.shape(gt_box_image)[0] == 0:
            continue

        num_gt_images = num_gt_images + 1
        
        if int(valid_detections[index].numpy()) == 0:
            if include_no_detection_as_zero:
                iou_incl0.append(tf.constant(0.0, tf.float32))
            continue

        # There are detections
        num_det_images = num_det_images + 1
        top_pred = top_boxes[index]
        
        iou_matrix = iou_matrix_core(top_pred, gt_box_image)
        best_iou = tf.reduce_max(iou_matrix)

        ious_only_det.append(best_iou)
        iou_incl0.append(best_iou)
        
    mean_only_detections = tf.reduce_mean(tf.stack(ious_only_det)) if len(ious_only_det) > 0 else tf.constant(0.0, tf.float32)
    mean_incl0 = tf.reduce_mean(tf.stack(iou_incl0)) if len(iou_incl0) > 0 else tf.constant(0.0, tf.float32)
    
    # There are IoU's that need to be averaged
    return {
        "mean_iou_top1_only_det": mean_only_detections,
        "mean_iou_top1_incl0": mean_incl0,
        "num_gt_images": tf.constant(num_gt_images, tf.int32),
        "num_det_images": tf.constant(num_det_images, tf.int32),
    }
    
# --- RUN UTILITIES --- #

def initialize_run_metadata(config: dict[str, Any], args: dict[str, Any], fingerprint: Fingerprint, timestamp: str):
    # Creating a fingerprint json representation
    fingerprint_json = {
        "fingerprint": fingerprint.hex,
        "schema_version": fingerprint.schema_version,
        "created_at": timestamp,
        "short": fingerprint.short
    }
    
    experiment_name = config.get("experiment", {}).get("id", "default_experiment")
    fingerprint_str = str(fingerprint.short) if fingerprint else "no_fingerprint"
    name_format = config.get("run", {}).get('name_format', "{experiment_id}_{fingerprint}_{timestamp}")
    
    name_format = name_format.replace("{experiment_id}", experiment_name)
    name_format = name_format.replace("{fingerprint}", fingerprint_str)
    name_format = name_format.replace("{timestamp}", timestamp)
    
    job_name = name_format
    run_dir = config.get("run", {}).get("root", "runs")
    
    # Creating the run fingerprint json metadata
    fingerprint_meta_dir = f"{run_dir}/{job_name}"
    
    # Creating a json file for the fingerprint
    fingerprint_json_path = f"{run_dir}/{job_name}/fingerprint.json"
    
    # Write to a file
    with open(fingerprint_json_path, 'w') as f:
        # Dumping the fingerprint json
        json.dump(fingerprint_json, f, indent=4)
        
    # Writing the config file
    config_json_path = f"{run_dir}/{job_name}/config.json"
    
    with open(config_json_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    # Writing a status file for the run
    status_json_path = f"{run_dir}/{job_name}/status.json"
    
    with open(status_json_path, 'w') as f:
        json.dump({"status": "initialized"}, f, indent=4)
        
    # Optionally, we could also write the args to a json file for better traceability
    args_json_path = f"{run_dir}/{job_name}/args.json"
    
    args_json = {key: str(value) if isinstance(value, Path) else value for key, value in args.items()}
    
    with open(args_json_path, 'w') as f:
        json.dump(args_json, f, indent=4)
    
