import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from typing import Any

class ExtraFeaturePyramid(tf.keras.layers.Layer):
    def __init__(self, name: str, extra_levels_cfg: list[dict[str,Any]], **kwargs):
        super().__init__(name=name, **kwargs)
        self.extra_heads_config = extra_levels_cfg
        self.level_names = []
        self.extra_heads = []
        
    def call(self,base_feature, training = False):
        x = base_feature
        extra_features = []

        for block in self.extra_heads:
            x = block(x,training = training)
            extra_features.append(x)

        return extra_features
        
    def build(self,input_shape: tf.Tensor):
        self.level_names = [config['name'] for config in self.extra_heads_config]
        
        for level, config in enumerate(self.extra_heads_config):
            out_channel = config['out_channels']
            blk_stride = config.get('stride',2)
            kernel_size = config.get('kernel_size',3)
            block = Conv2D(filters= out_channel, strides = blk_stride, kernel_size = kernel_size, padding= "same", activation="relu", name= f"extra_{level}_conv")

            self.extra_heads.append(block)

        super().build(input_shape)  