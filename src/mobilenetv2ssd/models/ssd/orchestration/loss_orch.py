import tensorflow as tf
from typing import Any

from mobilenetv2ssd.models.ssd.ops.loss_ops_tf import multibox_loss
from mobilenetv2ssd.core.precision_config import PrecisionConfig

def _extract_information_from_train_config(config : dict[str, Any]):
    loss_config = config.get("loss",{})
    loss_config = {
        "cls_loss_type": loss_config.get('classification').get("type","softmax_ce"),
        "reg_loss_type": loss_config.get('regression').get("type","smooth_l1"),
        "smooth_l1_beta": loss_config.get('regression').get("beta",1.0),
        "bbox_norm": loss_config.get("bbox_norm","none"),
        "from_logits": loss_config.get("from_logits",False),
        "ignore_index": loss_config.get("ignore_index",-1),
        "use_sigmoid": loss_config.get("use_sigmoid",False),
        "classification_weights": loss_config.get("cls_weight",1.0),
        "localization_weights": loss_config.get("reg_weight",1.0),
        "normalization_denom": loss_config.get("normalization",{}).get("type","num_pos"),
        "num_classes": config.get("num_classes",1),
        "reduction": loss_config.get("reduction","sum")
    }

    return loss_config

def calculate_final_loss(config: dict[str,Any], predicted_offsets: tf.Tensor, predicted_logits: tf.Tensor, localization_targets: tf.Tensor, classification_targets: tf.Tensor, positive_mask: tf.Tensor, negative_mask: tf.Tensor, precision_config : PrecisionConfig | None = None):
    # This is the function that calculates the multibox loss for the model
    # Steps:
    # 1. Get values from the config.
    # 2. Calculate the multibox loss

    loss_config = _extract_information_from_train_config(config)
    # loss_dict = tf.map_fn(lambda inputs: multibox_loss(predicted_offsets = inputs[0], predicted_logits = inputs[1], target_offsets = inputs[2], target_labels = inputs[3], positive_mask = inputs[4], negative_mask = inputs[5], localization_weight = loss_config["localization_weights"], classification_weight = loss_config["classification_weights"], beta = loss_config["smooth_l1_beta"], cls_loss_type = loss_config["cls_loss_type"],loc_loss_type = loss_config["reg_loss_type"], normalize_denom = loss_config["normalization_denom"], reduction = loss_config['reduction']),
    #                       elems = (predicted_offsets,predicted_logits,localization_targets,classification_targets,positive_mask,negative_mask),
    #                       fn_output_signature = {
    #                           "total_loss": tf.TensorSpec(shape=(), dtype=tf.float32),
    #                           "loc_loss": tf.TensorSpec(shape=(), dtype=tf.float32),
    #                           "cls_loss": tf.TensorSpec(shape=(), dtype=tf.float32),
    #                           "num_pos": tf.TensorSpec(shape=(), dtype=tf.int32),
    #                           "num_negative": tf.TensorSpec(shape=(), dtype=tf.int32)
    #                       }
    #                      )

    loss_dict = multibox_loss(predicted_offsets = predicted_offsets, predicted_logits = predicted_logits, target_offsets = localization_targets, target_labels = classification_targets, positive_mask = positive_mask, negative_mask = negative_mask, localization_weight = loss_config["localization_weights"], classification_weight = loss_config["classification_weights"], beta = loss_config["smooth_l1_beta"], cls_loss_type = loss_config["cls_loss_type"],loc_loss_type = loss_config["reg_loss_type"], normalize_denom = loss_config["normalization_denom"], reduction = loss_config['reduction'], precision_config= precision_config)

    return loss_dict