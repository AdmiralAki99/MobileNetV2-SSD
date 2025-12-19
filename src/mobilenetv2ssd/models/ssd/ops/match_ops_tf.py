import tensorflow as tf

from .box_ops_tf import cxcywh_toxyxy_core, iou_matrix_core
from mobilenetv2ssd.core.precision_config import PrecisionConfig, should_force_fp32

def _check_for_center_alignment(priors_cxcywh: tf.Tensor, gt_boxes_xyxy: tf.Tensor):
    
    cx,cy,_,_ = tf.split(priors_cxcywh,num_or_size_splits = 4, axis=-1)
    x_min, y_min, x_max, y_max = tf.split(gt_boxes_xyxy,num_or_size_splits = 4, axis=-1)

    cx = tf.transpose(cx, perm=[1, 0]) 
    cy = tf.transpose(cy, perm=[1, 0])
    
    # Check if (x_min <= cx <= x_max) and (y_min <= cy <= y_max)
    center_alignment = (x_min <= cx) & (cx<= x_max) & (y_min <= cy) & (cy <= y_max)

    return center_alignment

def _calculate_matches(iou_matrix: tf.Tensor,gt_boxes: tf.Tensor,positive_iou_thresh: float,negative_iou_thresh: float,enforce_bipartite: bool = True):
    # Apply rules
    M = tf.shape(gt_boxes)[0]
    N = tf.shape(iou_matrix)[1]
    
    max_iou_per_anchor = tf.reduce_max(iou_matrix,axis=0)
    assigned_gt_box_index = tf.argmax(iou_matrix,axis = 0,output_type = tf.int32)

    positive_mask = max_iou_per_anchor >= positive_iou_thresh
    negative_mask = max_iou_per_anchor < negative_iou_thresh
    ignore_mask = tf.logical_not(tf.logical_or(positive_mask,negative_mask))

    if enforce_bipartite:
        # For each GT box get the prior with the most value
        best_prior_per_gt = tf.argmax(iou_matrix, axis=1, output_type = tf.int32)

        # Get the values of the IoUs for the best matches
        best_iou_per_gt = tf.reduce_max(iou_matrix, axis=1)
        valid_gt = best_iou_per_gt > tf.constant(-0.5, iou_matrix.dtype)

        best_indices_all = tf.stack([tf.range(M, dtype=tf.int32), best_prior_per_gt], axis=1)  # (M,2)
        best_indices = tf.boolean_mask(best_indices_all, valid_gt)
        best_values  = tf.boolean_mask(tf.gather_nd(iou_matrix, best_indices_all), valid_gt)
        
        # Create a sparse Matrix to resolve any potential conflicts
        sparse_iou = tf.scatter_nd(best_indices,best_values,shape=tf.stack([M,N]))

        # Find which of the columns are forced
        forced_cols = tf.reduce_any(sparse_iou > tf.constant(-0.5, iou_matrix.dtype),axis=0)

        # Now calculate the best gt box per anchor to remove the conflicts by having the best one pick the prior
        best_gt_per_anchor = tf.argmax(sparse_iou, axis=0, output_type = tf.int32)

        # Now override the values where there was a force that was occuring
        resolved_gt_assignment = tf.where(forced_cols, best_gt_per_anchor, assigned_gt_box_index)

        # Now override the assignment for the  previous indices with the new resolved one
        assigned_gt_box_index = tf.where(forced_cols, resolved_gt_assignment, assigned_gt_box_index)

        # Stack the assigned gt box and each prior
        assigned_gt_box_per_prior = tf.stack([assigned_gt_box_index, tf.range(N,dtype=tf.int32)], axis=1)

        # Get the last resolved IoU matrix using it
        resolved_iou = tf.gather_nd(iou_matrix,assigned_gt_box_per_prior)
        max_iou_per_anchor = tf.where(forced_cols, resolved_iou, max_iou_per_anchor)

        # Now Update the masks with the new forced picks
        positive_mask = tf.logical_or(positive_mask,forced_cols)
        negative_mask = tf.where(forced_cols, tf.zeros_like(negative_mask),negative_mask)
        ignore_mask = tf.where(forced_cols, tf.zeros_like(ignore_mask),ignore_mask)
        
    
    # Calculate the number of positives
    number_of_positive_priors = tf.reduce_sum(tf.cast(positive_mask,tf.int32))
    # Calculate where the labels need to be ignored
    assigned_gt_box_index = tf.where(positive_mask, assigned_gt_box_index, -tf.ones_like(assigned_gt_box_index))

    # return assigned_gt_box_index, max_iou_per_anchor, positive_mask, negative_mask, ignore_mask, number_of_positive_priors
    return {
        "assigned_gt_box_index": assigned_gt_box_index,
        "max_iou_per_prior": max_iou_per_anchor,
        "pos_mask": positive_mask,
        "neg_mask": negative_mask,
        "ignore_mask": ignore_mask,
        "num_pos": number_of_positive_priors,
    }
    
def match_priors(priors_cxcywh: tf.Tensor, gt_boxes_xyxy: tf.Tensor, gt_labels: tf.Tensor, gt_valid_mask: tf.Tensor | None, positive_iou_thresh: float, negative_iou_thresh: float, max_pos_per_gt: list[int] | None = None, allow_low_qual_matches: bool = True, center_in_gt: bool = True , return_iou: bool = False, precision_config: PrecisionConfig | None = None):

    priors_cxcywh = tf.reshape(priors_cxcywh, [-1, 4])
    gt_boxes_xyxy = tf.reshape(gt_boxes_xyxy, [-1, 4])

    N = tf.shape(priors_cxcywh)[0]
    M = tf.shape(gt_boxes_xyxy)[0]

    # Fringe case, if gt_valid_mask is None then treat all boxes as valid
    if gt_valid_mask is None:
        gt_valid_mask = tf.ones_like(gt_labels,tf.bool)

    validity_check = tf.cast(gt_valid_mask,dtype=tf.int32)

    if tf.equal(tf.size(gt_boxes_xyxy),0) or tf.equal(tf.reduce_sum(validity_check),tf.constant(0,dtype=tf.int32)):
        return {
        "matched_gt_xyxy": tf.zeros([N, 4], tf.float32),
        "matched_gt_labels":  tf.zeros([N], tf.int32),
        "pos_mask":        tf.zeros([N], tf.bool),
        "neg_mask":        tf.ones([N],  tf.bool),
        "ignore_mask":     tf.zeros([N], tf.bool),
        "matched_gt_idx":  -tf.ones([N], tf.int32),
        "matched_iou":     tf.zeros([N], tf.float32),
        "num_pos":         tf.zeros([], tf.int32),
        }

    # Need to compute which of the gt boxes are valid
    valid_indices = tf.where(gt_valid_mask)

    valid_gt_boxes = tf.gather_nd(gt_boxes_xyxy, valid_indices)
    valid_labels = tf.gather_nd(gt_labels, valid_indices)

    # Compute the IoU Matrix
    priors_xyxy = cxcywh_toxyxy_core(priors_cxcywh)
    
    if should_force_fp32("iou",precision_config):
        valid_gt_boxes = tf.cast(valid_gt_boxes,tf.float32)
        priors_xyxy = tf.cast(priors_xyxy,tf.float32)
    
    iou_matrix = iou_matrix_core(valid_gt_boxes,priors_xyxy)

    if center_in_gt:
        center_aligned = _check_for_center_alignment(priors_cxcywh,valid_gt_boxes)
        # Filter out the Non centered priors
        iou_matrix = tf.where(center_aligned, iou_matrix, tf.zeros_like(iou_matrix))

    # Calculate matching mask
    match_dict = _calculate_matches(iou_matrix,valid_gt_boxes,positive_iou_thresh,negative_iou_thresh,enforce_bipartite = allow_low_qual_matches)

    labels_g = tf.gather(valid_labels,match_dict["assigned_gt_box_index"])
    boxes_g  = tf.gather(valid_gt_boxes, match_dict["assigned_gt_box_index"])

    zeros_labels = tf.zeros_like(match_dict["assigned_gt_box_index"], dtype=tf.int32)
    zeros_boxes  = tf.zeros([N, 4], dtype=gt_boxes_xyxy.dtype)
    
    # Calculate the matching labels
    matched_labels  = tf.where(match_dict["pos_mask"], labels_g, zeros_labels)

    # Calculate the matching ground truth boxes
    matched_gt_xyxy = tf.where(tf.expand_dims(match_dict["pos_mask"], 1), boxes_g, zeros_boxes)

    matched_gt_idx = tf.where(match_dict["pos_mask"], match_dict["assigned_gt_box_index"], -tf.ones_like(match_dict["assigned_gt_box_index"]))

    return_dict = {
        "matched_gt_xyxy" : matched_gt_xyxy,
        "matched_gt_labels": matched_labels,
        "pos_mask": match_dict['pos_mask'],
        "neg_mask": match_dict['neg_mask'],
        "ignore_mask": match_dict['ignore_mask'],
        "matched_gt_idx": matched_gt_idx,
        "num_pos": match_dict['num_pos']
    }

    if return_iou:
        # Calculate IoU for the images
        max_iou = match_dict["max_iou_per_prior"] 
        matched_iou = tf.where(match_dict["pos_mask"], max_iou, tf.zeros_like(max_iou))
        return_dict['matched_iou'] = matched_iou
    else:
        return_dict["matched_iou"] = tf.zeros([N], tf.float32)

    return return_dict

def hard_negative_mining(conf_loss: tf.Tensor, pos_mask: tf.Tensor, neg_mask:tf.Tensor, neg_ratio: float, min_neg: int| None, max_neg: int| None):
    
    num_positive = tf.reduce_sum(tf.cast(pos_mask, tf.int32))

    K = tf.math.floor(tf.cast(neg_ratio,dtype=tf.float32) * tf.cast(num_positive,dtype=tf.float32))
    K = tf.cast(K,tf.int32)

    if max_neg is not None:
        K = tf.minimum(K,tf.cast(max_neg,tf.int32))

    if min_neg is not None:
        K = tf.maximum(K,tf.cast(min_neg,tf.int32))

    K = tf.cast(tf.where(num_positive > 0, K, tf.zeros_like(K)),dtype=tf.int32)

    # Getting the indices for the negative boxes
    negative_indices = tf.where(neg_mask)[:,0]

    negative_losses = tf.gather(conf_loss,negative_indices)

    # Filtering the losses to not include NaN or inf
    valid_mask = tf.logical_not(tf.math.is_nan(negative_losses))
    valid_negative_indices = tf.boolean_mask(negative_indices,valid_mask)
    valid_negative_losses = tf.boolean_mask(negative_losses,valid_mask)

    # Checking if there are no valid losses
    num_valid_losses = tf.shape(valid_negative_losses)[0]
    k = tf.minimum(K,num_valid_losses)

    top_k_losses, top_k_indices = tf.math.top_k(valid_negative_losses,k=k,sorted=True)
    
    hard_negative_indices = tf.cast(tf.gather(valid_negative_indices,top_k_indices),tf.int32)

    hard_negative_indices = tf.expand_dims(hard_negative_indices,axis=1)

    selected_negative_mask = tf.scatter_nd(indices = hard_negative_indices, updates = tf.ones(k,dtype=tf.bool),shape=[tf.shape(conf_loss)[0]])

    selected_negative_indices = hard_negative_indices

    return selected_negative_mask, selected_negative_indices
