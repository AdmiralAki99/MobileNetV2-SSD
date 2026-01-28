import tensorflow as tf
from typing import Any

from mobilenetv2ssd.models.ssd.ops.match_ops_tf import hard_negative_mining

def _extract_information_from_train_config(config : dict[str, Any]):
    train_config = config['train']
    sampler_config = train_config.get("sampler",{})
    target_config = {
        "neg_pos_ratio": sampler_config.get("neg_pos_ratio",3.0),
        "min_neg":  sampler_config.get("min_neg",0),
        "max_neg": sampler_config.get("max_neg",None),
    }

    return target_config

def select_hard_negatives(config: dict[str, Any], conf_loss: tf.Tensor, positive_mask: tf.Tensor, negative_mask: tf.Tensor):
    # This function will orchestrate the hard negative mining that needs to be done
    # Steps:
    # 1. Get values from the config
    # 2. Calculate Hard negatives
    sampler_config = _extract_information_from_train_config(config)

    selected_negative_mask = tf.map_fn(lambda inputs: hard_negative_mining(conf_loss = inputs[0],pos_mask = inputs[1],neg_mask = inputs[2],neg_ratio = sampler_config['neg_pos_ratio'], min_neg = sampler_config['min_neg'], max_neg = sampler_config['max_neg']),
              elems = (conf_loss,positive_mask,negative_mask), 
              fn_output_signature = ( tf.TensorSpec(shape=(None,), dtype=tf.bool))
             )
         
    return selected_negative_mask