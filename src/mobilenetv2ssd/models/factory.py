import tensorflow as tf
from typing import Any

from mobilenetv2ssd.models.ssd.model import SSD

def _extract_information_from_train_config(config : dict[str, Any]):
    
    model_config = config['model']
    
    # Getting the specs for the model 
    input_shape = model_config.get('input_size',[300,300,3])
    alpha = model_config.get('width_mult',1.0)
    
    backbone_type = model_config.get('backbone','mobilenetv2')
    ssd_name = model_config.get('name',"mobilenetv2-ssd")
    
    num_classes = model_config['num_classes']
    feature_map_config = model_config['ssd']['feature_maps']
    
    backbone_features = feature_map_config.get('backbone_features', ["C3", "C4", "C5"])
    
    extra_base = feature_map_config.get('extra_base', None)
    
    extra_levels = feature_map_config.get('extra_levels', [{'name': 'P6', 'out_channels': 256, 'stride': 2, 'kernel_size': 3},{'name': 'P7', 'out_channels': 256, 'stride': 2, 'kernel_size': 3},{'name': 'P8', 'out_channels': 128, 'stride': 2, 'kernel_size': 3}])
    
    head_config = model_config.get("heads",{})
    localization_config = head_config.get('localization',{})
    classification_config = head_config.get('classification',{})

    return input_shape, alpha, backbone_type, ssd_name, num_classes, backbone_features, localization_config, classification_config,extra_levels, extra_base

def build_ssd_model(config: dict[str,Any], anchors_per_layer: list[int]):
    # Read from config
    input_shape, alpha, backbone_type, ssd_name, num_classes, backbone_features, localization_config, classification_config,extra_levels, extra_base = _extract_information_from_train_config(config)
    
    # Creating the SSD model
    ssd = SSD(backbone_type = backbone_type, name = ssd_name, feature_maps = backbone_features, number_of_classes = num_classes, number_of_anchors_per_layer = anchors_per_layer, input_shape = tuple(input_shape), loc_head_configuration =localization_config , cls_head_configuration = classification_config, extra_levels = extra_levels, extra_base = extra_base, alpha = alpha)

    dummy_shape = [1] + input_shape

    dummy_image = tf.random.uniform(dummy_shape, dtype=tf.float32)

    ssd(dummy_image)

    return ssd