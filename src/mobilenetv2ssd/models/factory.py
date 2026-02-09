import tensorflow as tf
from typing import Any

from mobilenetv2ssd.models.ssd.model import SSD

def _extract_information_from_model_config(config : dict[str, Any]):
       
    # Getting the specs for the model 
    input_shape = config['input_size'] + [3]
    alpha = config.get('backbone',{}).get('width_mult',1.0)
    
    backbone_type =  config.get('backbone',{}).get('name','mobilenetv2')
    ssd_name = backbone_type + '-ssd'
    
    num_classes = config['num_classes']
    
    backbone_features = config.get('backbone',{}).get('output_layers',["C3", "C4", "C5"])
    
    extra_base = config.get('heads',{}).get('extra_layers',{}).get('base_feature',backbone_features[-1])
    
    extra_levels =  config.get('heads',{}).get('extra_layers',{}).get('levels',[{'name': 'P6', 'out_channels': 256, 'stride': 2, 'kernel_size': 3},{'name': 'P7', 'out_channels': 256, 'stride': 2, 'kernel_size': 3},{'name': 'P8', 'out_channels': 128, 'stride': 2, 'kernel_size': 3}])
    
    localization_config = config.get("heads",{}).get('localization',{})
    classification_config = config.get("heads",{}).get('classification',{})

    return input_shape, alpha, backbone_type, ssd_name, num_classes, backbone_features, localization_config, classification_config,extra_levels, extra_base

def build_ssd_model(config: dict[str,Any], anchors_per_layer: list[int]):
    # Read from config
    input_shape, alpha, backbone_type, ssd_name, num_classes, backbone_features, localization_config, classification_config,extra_levels, extra_base = _extract_information_from_model_config(config)
    
    # Creating the SSD model
    ssd = SSD(backbone_type = backbone_type, name = ssd_name, feature_maps = backbone_features, number_of_classes = num_classes, number_of_anchors_per_layer = anchors_per_layer, input_shape = tuple(input_shape), loc_head_configuration =localization_config , cls_head_configuration = classification_config, extra_levels = extra_levels, extra_base = extra_base, alpha = alpha)

    dummy_shape = [1] + input_shape

    dummy_image = tf.random.uniform(dummy_shape, dtype=tf.float32)

    ssd(dummy_image)

    return ssd