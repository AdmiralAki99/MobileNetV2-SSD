import numpy as np
import tensorflow as tf
import pytest
from model.mobilenetv2_backbone import (build_mobilenetv2_backbone, load_mobilenetv2_weights)


def max_abs_diff(a, b):
    return float(tf.reduce_max(tf.abs(a - b)).numpy())

@pytest.fixture(scope="module")
def ref_model():
    return tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet")

@pytest.fixture(scope="module")
def my_model():
    # build your model and load transplanted weights here
    model = build_mobilenetv2_backbone()
    load_mobilenetv2_weights(model)
    return model

def assert_weights_equal(ref_layer, my_layer):
    s, d = ref_layer.get_weights(), my_layer.get_weights()
    assert len(s) == len(d), f"Var count mismatch: {ref_layer.name} vs {my_layer.name}"
    for i, (sw, dw) in enumerate(zip(s, d)):
        assert sw.shape == dw.shape, f"Shape mismatch at {ref_layer.name}[{i}]"
        assert np.array_equal(sw, dw), f"Weights differ at {ref_layer.name}[{i}]"
        
def test_forward_equivalence_end_to_end(ref_model, my_model):
    tf.random.set_seed(0)
    x = tf.random.uniform([1, 224, 224, 3], dtype=tf.float32)
    y_ref = ref_model(x, training=False)
    y_my  = my_model(x, training=False)
    diff = max_abs_diff(y_ref, y_my)
    assert diff <= 1e-5, f"End-to-end diff too large: {diff}"