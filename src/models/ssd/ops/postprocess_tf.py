import tensorflow as tf

def _decode_boxes(predicted_offsets: tf.Tensor, priors: tf.Tensor, variances: tf.Tensor):
    B = tf.shape(predicted_offsets)[0]
    N = tf.shape(predicted_offsets)[1]

    # Variance size
    variance_center = variances[0]
    variance_shape = variances[1]
    
    # Broadcasting the priors to the shape for the offsets
    broadcasted_priors = tf.broadcast_to(priors[tf.newaxis,...],[B,N,4])

    # Decoding the boxes
    tx,ty,tw,th = tf.split(predicted_offsets,num_or_size_splits = 4, axis=-1)
    cx,cy,w,h = tf.split(broadcasted_priors,num_or_size_splits = 4, axis=-1)

    # Adding the offset to the coordinates
    cx = cx + tx * variance_center * w
    cy = cy + ty * variance_center * h
    w = w * tf.math.exp(tw * variance_shape)
    h = h * tf.math.exp(th * variance_shape)

    # Converting the boxes to xy-coordinates
    x_min = cx - w/2
    y_min = cy - h/2
    x_max = cx + w/2
    y_max = cy + h/2

    boxes_xyxy = tf.concat([x_min, y_min, x_max, y_max],axis=-1)

    boxes_xyxy = tf.clip_by_value(boxes_xyxy,0,1)

    return boxes_xyxy

def _softmax_probabilities(logits: tf.Tensor):
    logits = tf.cast(logits, tf.float32)

    max_per_row = tf.reduce_max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_per_row

    exponential_shifted_logits = tf.math.exp(shifted_logits)
    sum_exponential_shifted_logits = tf.reduce_sum(exponential_shifted_logits, axis=-1, keepdims=True)

    softmax_probs = exponential_shifted_logits / sum_exponential_shifted_logits

    return softmax_probs

def _sigmoid_probabilities(logits: tf.Tensor):
    logits = tf.cast(logits, tf.float32)

    sigmoid_probabilities = 1/(1 + tf.math.exp(-logits))

    return sigmoid_probabilities
    

def _score_from_logits(predicted_logits: tf.Tensor, scores_thresh: float, use_sigmoid: bool = True):
    if use_sigmoid:
        sigmoid_probs = _sigmoid_probabilities(predicted_logits)

        thresh_mask = sigmoid_probs < scores_thresh

        thresholded_scores = tf.where(thresh_mask, tf.zeros_like(sigmoid_probs), sigmoid_probs)
    else:
        softmax_probs = _softmax_probabilities(predicted_logits)
        scores = softmax_probs[...,1:]

        # Applying the threshold
        thresh_mask = scores < scores_thresh

        # Updating the scores in the tensor
        thresholded_scores = tf.where(thresh_mask, tf.zeros_like(scores), scores)
    
    return thresholded_scores

def _prepare_nms_inputs(boxes_xyxy: tf.Tensor, scores: tf.Tensor):
    nms_boxes = boxes_xyxy[:,:,tf.newaxis,:]
    nms_scores = scores
    
    return nms_boxes,nms_scores

def _run_batched_nms(nms_boxes: tf.Tensor, nms_scores: tf.Tensor, iou_thresh: float, scores_thresh: float, top_k: int, max_detections: int):
    
    nms_boxes, nms_scores, nms_classes, valid_detections = tf.image.combined_non_max_suppression(nms_boxes,nms_scores,top_k,max_detections,iou_threshold= iou_thresh,score_threshold= scores_thresh,clip_boxes=False)

    return nms_boxes,nms_scores,nms_classes,valid_detections

def _restore_to_image_space(boxes_yxyx_norm: tf.Tensor, image_height: int, image_width: int):
    # Multiplying the normalized coordinates with the 
    y_min,x_min,y_max,x_max = tf.split(boxes_yxyx_norm,num_or_size_splits = 4, axis=-1)

    # Scaling the box back to the image
    y_min = y_min * image_height
    x_min = x_min * image_width

    y_max = y_max * image_height
    x_max = x_max * image_width

    # Stacking the values back
    return tf.concat([x_min,y_min,x_max,y_max], axis=-1)

def _filter_small_boxes(boxes_norm_xyxy: tf.Tensor, scores:tf.Tensor, min_size: float):

    # Safety check
    if min_size is None or min_size <= 0:
         return boxes_norm_xyxy, scores, tf.ones(tf.shape(boxes_norm_xyxy)[:2], dtype=tf.bool)
    
    # Splitting the coordinates
    x_min,y_min,x_max,y_max = tf.split(boxes_norm_xyxy,num_or_size_splits = 4, axis=-1)

    # Calculating the width and height
    width = x_max - x_min
    height = y_max - y_min

    # Masking stuff to keep
    keep_mask = (width >= min_size) & (height >= min_size)

    # Gathering the box anchors
    gathered_anchors = tf.where(keep_mask, boxes_norm_xyxy, tf.zeros_like(boxes_norm_xyxy))

    # Gathering the scores
    gathered_scores = tf.where(keep_mask, scores, tf.zeros_like(scores))

    return gathered_anchors, gathered_scores, keep_mask

def _pre_nms_top_k(boxes_xyxy: tf.Tensor, scores: tf.Tensor, top_k: int):

    tf.debugging.assert_equal(tf.shape(boxes_xyxy)[-1], 4)
    tf.debugging.assert_equal(tf.shape(boxes_xyxy)[1], tf.shape(scores)[1])

    boxes_xyxy = tf.cast(boxes_xyxy,tf.float32)
    scores = tf.cast(scores,tf.float32)

    N = tf.shape(boxes_xyxy)[1]

    top_k = tf.minimum(N,top_k)

    # Max scores
    max_scores = tf.reduce_max(scores,axis=-1)

    # Get top k scores
    _, top_k_indices = tf.math.top_k(max_scores,k=top_k)

    # Get the top k boxes
    boxes_top_k = tf.gather(boxes_xyxy,top_k_indices, batch_dims = 1)

    # Gathering all the scores associated with the indices
    scores_top_k = tf.gather(scores,top_k_indices, batch_dims = 1)

    return boxes_top_k, scores_top_k

def decode_and_nms(predicted_offsets: tf.Tensor, predicted_logits: tf.Tensor, priors: tf.Tensor, variances: tf.Tensor, scores_thresh: float, iou_thresh: float, top_k: int, max_detections: int, image_meta: dict| None, use_sigmoid: bool = False, **kwargs):
    # Decoding the boxes
    boxes_xyxy = _decode_boxes(predicted_offsets = predicted_offsets, priors = priors, variances = variances)

    # Calculating the scores from the logits
    scores = _score_from_logits(predicted_logits = predicted_logits, scores_thresh = scores_thresh, use_sigmoid = use_sigmoid)

    # Check if there is a filter option
    if 'min_box_size' in kwargs:
       boxes_xyxy, scores, keep_mask = _filter_small_boxes(boxes_xyxy, scores,kwargs['min_box_size'])

    if 'pre_nms_top_k' in kwargs:
        boxes_xyxy, scores = _pre_nms_top_k(boxes_xyxy, scores, kwargs['pre_nms_top_k'])

    # Preparing the inputs for NMS outputs
    nms_boxes, nms_scores = _prepare_nms_inputs(boxes_xyxy, scores)

    # Run batched NMS
    nmsed_boxes,nmsed_scores, nmsed_classes, valid_detections = _run_batched_nms(nms_boxes,nms_scores,iou_thresh, scores_thresh, top_k, max_detections)

    if not use_sigmoid:
        # Accounting for Softmax probs
        nmsed_classes = tf.cast(nmsed_classes,tf.int32)
        nmsed_classes = nmsed_classes + 1

    # Returning the boxes to image space
    if image_meta is not None:
        nmsed_boxes = _restore_to_image_space(nmsed_boxes,image_meta['image_height'], image_meta['image_width'])
    
    return nmsed_boxes,nmsed_scores, nmsed_classes, valid_detections