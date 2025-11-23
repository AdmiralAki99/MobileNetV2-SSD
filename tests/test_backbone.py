import tensorflow as tf
from mobilenetv2ssd.models.mobilenet_v2.backbone import *

import pytest

# Testing the creation of the translation block

def test_creating_translation_block():
    translation_block_2 =  {
        f"bottleneck_block_2_expand_conv" : f"block_1_expand",
        f"bottleneck_block_2_expand_bn"   : f"block_1_expand_BN",
        f"bottleneck_block_2_expand_relu6": f"block_1_expand_relu",
        f"bottleneck_block_2_dw_conv"     : f"block_1_depthwise",
        f"bottleneck_block_2_dw_bn"       : f"block_1_depthwise_BN",
        f"bottleneck_block_2_dw_relu6"    : f"block_1_depthwise_relu",
        f"bottleneck_block_2_project_conv": f"block_1_project",
        f"bottleneck_block_2_project_bn"  : f"block_1_project_BN",
    }
    
    block_2 = make_block_map(2,1)
    
    assert translation_block_2 == block_2, "Block 2 did not have the same layers"
    
def test_model_transplant():
    input_shape = (224,224,3)
    tensor_shape = [1, 224, 224, 3]
    alpha = 1.0
    model = build_backbone(input_shape,alpha=alpha)
    ref_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                  alpha=alpha,
                                                  include_top=False,
                                                  weights='imagenet',
                                                  input_tensor=None,
                                                  pooling=None,
                                                  classes=1000,
                                                  classifier_activation='softmax')
    
    x =  tf.random.uniform(tensor_shape, dtype=tf.float32)
    
    y_ref = ref_model(x, training=False)
    y_mine = model(x, training=False)
    
    tf.debugging.assert_near(y_ref, y_mine)
    