import tensorflow as tf
from typing import Any

from mobilenetv2ssd.models.ssd.ops.heads_tf import *
from mobilenetv2ssd.models.mobilenet_v2.backbone import build_backbone
from mobilenetv2ssd.models.ssd.fpn import ExtraFeaturePyramid

class SSD(tf.keras.Model):
    def __init__(self, backbone_type: str, name: str, feature_maps: list[str], number_of_classes: int, number_of_anchors_per_layer: list[int], input_shape: tuple[int,int,int], loc_head_configuration: dict[str, Any],cls_head_configuration: dict[str, Any],extra_levels: list[dict[str,Any]], extra_base: str | None, alpha: float = 1.0 ,**kwargs):
        super().__init__(name=name, **kwargs)
        
        self.feature_maps = feature_maps
        self.backbone = build_backbone(input_shape = input_shape,alpha = alpha, name = backbone_type)
        
        # Initializing the config for the extra heads
        self.extra_pyramid = ExtraFeaturePyramid(name = "extra_pyramid", extra_levels_cfg = extra_levels)
        self.extra_base = feature_maps[-1] if extra_base is None else extra_base

        # Localization and Classification Heads
        self.localization_head = LocalizationHead(name = "loc_head", num_anchors_per_location = number_of_anchors_per_layer, head_type = loc_head_configuration.get("head_type","conv3x3"), initial_norm_strategy = loc_head_configuration.get("initial_norm_strategy","BatchNorm"), squeeze_ratio = loc_head_configuration.get("squeeze_ratio",1.0), intermediate_conv = loc_head_configuration.get("intermediate_conv",128), in_channels =  loc_head_configuration.get("in_channels",[256,512,512])) 
        self.classification_head = ClassificationHead(name = "loc_head", num_anchors_per_location = number_of_anchors_per_layer, number_of_classes = number_of_classes , head_type = cls_head_configuration.get("head_type","conv3x3"), norm_cfg = cls_head_configuration.get("initial_norm_strategy","BatchNorm"), squeeze_ratio = cls_head_configuration.get("squeeze_ratio",1.0), intermediate_conv = cls_head_configuration.get("intermediate_conv",128))
        
    
    def call(self,image: tf.Tensor,training = False):
        
        feature_map_dict = self.backbone(image, training = training)

        feature_maps = [feature_map_dict[key] for key in self.feature_maps]

        # Getting the base feature for the pyramid
        base = feature_map_dict[self.extra_base]

        # Passing it through the extra pyramid
        extra_features = self.extra_pyramid(base,training = training)
        all_features = feature_maps + extra_features

        # Pass through the localization and classification heads
        pred_offsets = self.localization_head(all_features, training = training)
        pred_logits = self.classification_head(all_features, training = training)

        return pred_offsets, pred_logits

    def build(self,input_shape):

        # Building the model
        dummy_image = tf.zeros((1,) + tuple(input_shape[1:])) 
        feature_dict = self.backbone(dummy_image,training = False)

        # Getting the base feature
        feature_maps = [feature_dict[key] for key in self.feature_maps]
        base = feature_dict[self.extra_base]

        # Passing it through the Extra Pyramid
        extra_features = self.extra_pyramid(base,training = False)
        all_features = feature_maps + extra_features

        # Pass through the localization and classification heads
        _ = self.localization_head(all_features, training = False)
        _ = self.classification_head(all_features, training = False)

        super().build(input_shape)
    