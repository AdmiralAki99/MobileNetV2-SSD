import tensorflow as tf

from .box_ops_tf import xyxy_to_cxcywh_core
from mobilenetv2ssd.core.precision_config import PrecisionConfig, should_force_fp32

def _sanitize_boxes_xyxy(boxes_normalized: tf.Tensor):
    # Making sure the format of the boxes is correct
    x_min, y_min, x_max, y_max = tf.split(boxes_normalized,num_or_size_splits = 4, axis=-1)

    # Making sure the coordinate relationship is maintained
    x_min_filtered = tf.minimum(x_min,x_max)
    y_min_filtered = tf.minimum(y_min,y_max)
    x_max_filtered = tf.maximum(x_min,x_max)
    y_max_filtered = tf.maximum(y_min,y_max)

    # Making sure the values are normalized
    x_min_clipped = tf.clip_by_value(x_min_filtered,0,1)
    y_min_clipped = tf.clip_by_value(y_min_filtered,0,1)
    x_max_clipped = tf.clip_by_value(x_max_filtered,0,1)
    y_max_clipped = tf.clip_by_value(y_max_filtered,0,1)

    return tf.concat([x_min_clipped,y_min_clipped,x_max_clipped,y_max_clipped],axis=-1)

def encode_boxes_core(gt_boxes_xyxy: tf.Tensor, priors_cxcywh: tf.Tensor, variance: tuple[float,float], precision_config: PrecisionConfig | None = None):
    # Convert boxes to center coordinates
    if should_force_fp32("box_encode_decode", precision_config):
        gt_boxes_xyxy = tf.cast(gt_boxes_xyxy,tf.float32)
        priors_cxcywh = tf.cast(priors_cxcywh,tf.float32)
    
    gt_boxes_cxcywh = xyxy_to_cxcywh_core(gt_boxes_xyxy)
    
    gt_xc, gt_yc, gt_w, gt_h = tf.split(gt_boxes_cxcywh,num_or_size_splits = 4, axis=-1)

    prior_xc, prior_yc, prior_w, prior_h = tf.split(priors_cxcywh,num_or_size_splits = 4, axis=-1)

    variance_center = variance[0]
    variance_size = variance[1]

    # Protecting a division by zero to be inf or -inf
    eps = tf.constant(1e-8, dtype=gt_boxes_xyxy.dtype)
    gt_w = tf.maximum(eps,gt_w)
    gt_h = tf.maximum(eps,gt_h)
    prior_w = tf.maximum(eps,prior_w)
    prior_h = tf.maximum(eps,prior_h)

    # Calculate the offsets using the formulae from the paper
    tx = ((gt_xc - prior_xc)/prior_w) / variance_center
    ty = ((gt_yc - prior_yc)/prior_h) / variance_center
    tw = (tf.math.log(gt_w/prior_w)) / variance_size
    th = (tf.math.log(gt_h/prior_h)) / variance_size

    offsets = tf.concat([tx,ty,tw,th], axis= -1)

    # Calculate where the gt_boxes are padded
    padded_mask = tf.reduce_all(gt_boxes_xyxy == 0.0, axis=-1)

    update_mask = tf.expand_dims(padded_mask,axis=-1)

    # Where the mask is True, the GT box is padded there so to return the value to be 0.0, everywhere else keep it as is
    offsets = tf.where(update_mask,tf.zeros_like(offsets),offsets)

    return  offsets

def encode_boxes_batch(matched_gt_xyxy: tf.Tensor,priors_cxcywh: tf.Tensor, variances: tuple[float,float], precision_config: PrecisionConfig | None = None):
    # Need to create a function that encodes boxes by batch
    B = tf.shape(matched_gt_xyxy)[0]

    # Map over the entire batch
    batched_offsets = tf.map_fn(lambda matched_boxes: encode_boxes_core(matched_boxes,priors_cxcywh,variances,precision_config), elems=matched_gt_xyxy,fn_output_signature=tf.TensorSpec(shape=(None, 4), dtype=tf.float32))

    return batched_offsets