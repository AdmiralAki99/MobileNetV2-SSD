import tensorflow as tf
from typing import Any

from mobilenetv2ssd.models.ssd.ops.loss_ops_tf import softmax_cross_entropy_loss

def _extract_information_from_train_config(config : dict[str, Any]):
    train_config = config['train']
    loss_config = train_config.get("loss",{})
    loss_config = {
        "cls_loss_type": loss_config.get("cls_loss_type","ce_softmax"),
        "from_logits": loss_config.get("from_logits",False),
        "ignore_index": loss_config.get("ignore_index",-1),
        "use_sigmoid": loss_config.get("use_sigmoid",False),
    }

    return loss_config

def build_conf_loss(config: dict[str,Any], predicted_logits: tf.Tensor, classification_targets: tf.Tensor, pos_mask: tf.Tensor, neg_mask: tf.Tensor, ignore_mask: tf.Tensor):
    # This is the orchestrator to calculate the confidence loss between the priors and the matched boxes
    # Steps:
    # 1. Get the config for the loss
    # 2. Calculate the valid mask (Safeguard, my model should remove it already)
    # 3. Calculate the classification loss per anchor
    # 4. Apply label smoothing (Optional Implementation)
    # 5. Multiply Class wieights (Optional Implementation)
    
    loss_config = _extract_information_from_train_config(config)

    candidate_negative_mask = tf.logical_and(neg_mask,tf.logical_not(ignore_mask))
    ignored_labels = tf.zeros_like(classification_targets)

    valid_labels = tf.where(tf.logical_not(ignore_mask),classification_targets,ignored_labels)

    if loss_config['cls_loss_type'] == 'softmax_ce':
        per_class_loss = tf.map_fn(lambda inputs: softmax_cross_entropy_loss(inputs[0],inputs[1],reduction="none"),
                 elems = (predicted_logits, valid_labels),
                 fn_output_signature = tf.TensorSpec(shape=(None,), dtype=tf.float32)
                 )


    # Handle NaN
    nan = tf.constant(float("nan"), dtype = per_class_loss.dtype)

    conf_loss = tf.where(tf.logical_not(ignore_mask), per_class_loss, tf.fill(tf.shape(per_class_loss), nan))


    return conf_loss, candidate_negative_mask