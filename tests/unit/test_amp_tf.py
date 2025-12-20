import tensorflow as tf
import math

from training.amp import AMPContext
from mobilenetv2ssd.models.ssd.ops.encode_ops_tf import encode_boxes_core

import pytest
pytestmark = pytest.mark.unit

def test_setup_policy_sets_expected_global_policy():
    # Reset to a known baseline to avoid cross-test contamination
    tf.keras.mixed_precision.set_global_policy("float32")

    opt = tf.keras.optimizers.Adam(1e-3)
    config = {
        'enabled': True,
        'policy': 'mixed_float16',
        'loss_scale': 'dynamic',
        'clip_unscaled_grads': True,
        'force_fp32': ['loss_reduction', 'box_encode_decode', 'iou', 'nms']
    }
    amp = AMPContext(config= config, optimizer= opt)

    amp.setup_policy()

    global_policy_name = tf.keras.mixed_precision.global_policy().name
    assert global_policy_name == "mixed_float16"
    
def test_optimizer_is_wrapped_when_amp_enabled():
    
    tf.keras.mixed_precision.set_global_policy("float32")
    
    opt = tf.keras.optimizers.Adam(1e-3)
    config = {
        'enabled': True,
        'policy': 'mixed_float16',
        'loss_scale': 'dynamic',
        'clip_unscaled_grads': True,
        'force_fp32': ['loss_reduction', 'box_encode_decode', 'iou', 'nms']
    }
    
    amp = AMPContext(config= config, optimizer= opt)
    
    amp.setup_policy()
    
    amp.wrap_optimizer()
    
    assert amp.optimizer is not opt
    
def test_optimizer_is_not_wrapped_when_amp_disabled():
    
    opt = tf.keras.optimizers.Adam(1e-3)
    
    config = {
        'enabled': False,
        'policy': 'mixed_float16',
        'loss_scale': 'dynamic',
        'clip_unscaled_grads': True,
        'force_fp32': ['loss_reduction', 'box_encode_decode', 'iou', 'nms']
    }
    
    amp = AMPContext(config= config, optimizer= opt)

    amp.setup_policy()
    
    amp.wrap_optimizer()
    
    assert amp.optimizer is opt
    
def test_autocast_changes_compute_dtype_on_gpu_or_policy_on_any_device():
    
    tf.keras.mixed_precision.set_global_policy("float32")
   
    opt = tf.keras.optimizers.Adam(1e-3)
    
    config = {
        'enabled': True,
        'policy': 'mixed_float16',
        'loss_scale': 'dynamic',
        'clip_unscaled_grads': True,
        'force_fp32': ['loss_reduction', 'box_encode_decode', 'iou', 'nms']
    }
    
    amp = AMPContext(config= config, optimizer= opt)
    
    amp.setup_policy()
    
    amp.wrap_optimizer()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4),
    ])
    
    assert tf.keras.mixed_precision.global_policy().name == "mixed_float16"
    
    if len(tf.config.list_physical_devices("GPU")) > 0:
        assert model.layers[0].dtype_policy.compute_dtype == "float16"
    else:
        assert model.layers[0].dtype_policy is not None
        
        
def test_one_train_step_grads_finite_and_weights_update_with_amp():
    
    tf.keras.mixed_precision.set_global_policy("float32")
   
    opt = tf.keras.optimizers.Adam(1e-3)
    
    config = {
        'enabled': True,
        'policy': 'mixed_float16',
        'loss_scale': 'dynamic',
        'clip_unscaled_grads': True,
        'force_fp32': ['loss_reduction', 'box_encode_decode', 'iou', 'nms']
    }
    
    amp = AMPContext(config= config, optimizer= opt)
    
    amp.setup_policy()
    
    amp.wrap_optimizer()
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4),
    ])
    
    x = tf.random.normal([8, 16])
    y = tf.random.normal([8, 4])
    
    w0_before = tf.identity(model.trainable_variables[0])
    
    with tf.GradientTape() as tape:
        with amp.autocast():
            y_pred = model(x, training=True)
            y_pred = tf.cast(y_pred, tf.float32)
            y_true = tf.cast(y, tf.float32)
            loss = tf.reduce_mean(tf.square(y_pred - y_true))
            
    grads = tape.gradient(loss, model.trainable_variables)
    
    none_count = sum(g is None for g in grads)
    
    assert none_count == 0, f"Found {none_count} None gradients."
    
    grads = amp.scale_loss(grads)
    
    global_norm = tf.linalg.global_norm(grads)
    
    assert tf.math.is_finite(global_norm), "Gradient norm is NaN/Inf."
    
    assert float(global_norm.numpy()) > 0.0, "Gradient norm is zero (unexpected for random data)."
    
    amp.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    w0_after = model.trainable_variables[0]
    
    delta = tf.reduce_sum(tf.abs(tf.cast(w0_after, tf.float32) - tf.cast(w0_before, tf.float32)))
    
    assert float(delta.numpy()) > 0.0, "Weights did not change after apply_gradients()."
    

def test_check_mixed_precision():
    
    tf.keras.mixed_precision.set_global_policy("float32")
   
    opt = tf.keras.optimizers.Adam(1e-3)
    
    config = {
        'enabled': True,
        'policy': 'mixed_float16',
        'loss_scale': 'dynamic',
        'clip_unscaled_grads': True,
        'force_fp32': ['loss_reduction', 'box_encode_decode', 'iou', 'nms']
    }
    
    amp = AMPContext(config= config, optimizer= opt)
    
    amp.setup_policy()
    
    amp.wrap_optimizer()
    
    precision_config = amp.make_precision_config()
    
    priors_cxcywh = tf.constant([0.5, 0.5, 0.2, 0.2], dtype = tf.float32)
    gt_xyxy =  tf.constant([
        [0.10, 0.10, 0.30, 0.30],  # GT 0  (class e.g. 3)
        [0.55, 0.55, 0.85, 0.85],  # GT 1  (class e.g. 2)
        [0.20, 0.50, 0.40, 0.80],  # GT 2  (class e.g. 5)
        [0.00, 0.00, 0.00, 0.00],  # padded
    ], dtype=tf.float32)
    variance = (0.1,0.2)
    
    encode_boxes_core(gt_xyxy,priors_cxcywh,variance, precision_config = precision_config)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(16,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4),
    ])
    
    x = tf.random.normal([8, 16])
    y = tf.random.normal([8, 4])
    
    with tf.GradientTape() as tape:
        with amp.autocast():
            y_pred = model(x, training=True)
            
    tf.keras.mixed_precision.global_policy().name
    
    assert type(amp.optimizer) == tf.keras.mixed_precision.LossScaleOptimizer, f"AMP wrapped assert failed expected {tf.keras.optimizers.loss_scale_optimizer.LossScaleOptimizer} for {type(amp.optimizer)} "
    
    assert y_pred.dtype == tf.float16, f"AMP predicition dtype mismatch, expected: {tf.float16}, got {y_pred.dtype}"
    
    assert model.trainable_variables[0].dtype == tf.float32, f"AMP predicition dtype mismatch, expected: {tf.float32}, got {model.trainable_variables[0].dtype}"
    
    
    
    
    
    
    