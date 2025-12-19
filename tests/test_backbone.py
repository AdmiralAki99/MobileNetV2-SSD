import tensorflow as tf
import math

from mobilenetv2ssd.models.mobilenet_v2.backbone import *

## TODO: Integrate new changes

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
    
def test_build_model():
    input_shape = (224,224,3)
    alpha = 1.0
    model = MobileNetV2(alpha= alpha)
    
    assert isinstance(model, MobileNetV2), "MobileNetV2 build model failed with the wrong instance type"

def test_smoke_forward_pass():
    # Checking if the wiring of the blocks is done correctly
    input_shape = (224,224,3)
    alpha = 1.0
    model = MobileNetV2(alpha= alpha)
    
    # Expected keys from the model
    expected_keys = {"C2","C3","C4","C5"}
    
    # Building test input
    tensor_shape = [1, 224, 224, 3]
    x =  tf.random.uniform(tensor_shape, dtype=tf.float32)
    
    # Passing it through the model
    
    output = model(x, training = False)
    
    # Asserts
    assert output.keys() == expected_keys, "The keys from the model output were different"
    
    for key in expected_keys:
        assert output[key].ndim == 4, f"The dimensions for {key} are wrong: expected 4, got {output[key].ndim}"
        
    for key in expected_keys:
        assert output[key].shape[0] == tensor_shape[0], f"The batch dimension for {key} are wrong: expected {tensor_shape[0]}, got {output[key].shape[0]}"
        

def test_spatial_downscaling():
    # Checking if the downscaling for the feature maps is being done correctly
    input_shape = (224,224,3)
    alpha = 1.0
    model = MobileNetV2(alpha= alpha)
    
    # Expected keys from the model and their corresponding downscaling factor
    expected_downscaling = {"C2": 4,"C3" : 8,"C4": 16,"C5": 32}
    
    # Building test input
    tensor_shape = [1, 224, 224, 3]
    x =  tf.random.uniform(tensor_shape, dtype=tf.float32)
    
    output = model(x, training = False)
    
    # Asserts
    for key in expected_downscaling.keys():
        assert math.ceil(input_shape[0]/output[key].shape[1]) == expected_downscaling[key]  and math.ceil(input_shape[1]/output[key].shape[2]) == expected_downscaling[key], f"The downsampling for {key} is wrong: expected {expected_downscaling[key]}, got {(math.ceil(input_shape[0]/output[key][1]), math.ceil(input_shape[1]/output[key][2]))}"
    
    
def test_model_scaling():
    test_alpha = [1, 0.5, 1.4]
    
    # Building test input
    tensor_shape = [1, 224, 224, 3]
    x =  tf.random.uniform(tensor_shape, dtype=tf.float32)
    
    expected_channels = {'1.0': [24, 32, 96, 1280], '0.5': [16, 16, 48, 1280], '1.4': [32, 48, 136, 1792]}
    observed_channels = {}
    
    for alpha in test_alpha:
        model = MobileNetV2(alpha= alpha)
        output = model(x, training = False)
        
        c2, c3, c4, c5 = [output[key].shape[-1] for key in ["C2","C3","C4","C5"]]
        
        observed_channels[str(float(alpha))] = [c2, c3, c4, c5]
        
    for alpha in expected_channels.keys():
        assert expected_channels[alpha] == observed_channels[alpha], f"Unexpected channel scaling for {alpha}: expected {expected_channels[alpha]}, got {observed_channels[alpha]}"
  
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
    
    tf.debugging.assert_near(y_ref, y_mine['C5'])
    

def test_model_determinism():
    tensor_shape = [1, 224, 224, 3]
    alpha = 1.0
    model = MobileNetV2(alpha= alpha)
    
    x =  tf.random.uniform(tensor_shape, dtype=tf.float32)
    
    output = model(x, training = False)
    
    output_2 = model(x, training = False)
    
    for key in output.keys():
        tf.debugging.assert_equal(output[key],output_2[key], message= f"Determinism failed for {key}")