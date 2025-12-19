import tensorflow as tf
import pytest
import string

from mobilenetv2ssd.models.ssd.ops.anchor_ops_tf import *

# Disable eager execution for testing purposes
# tf.compat.v1.disable_eager_execution()

# Test cases for calculate_feature_map_shapes function
def test_calculate_feature_map_shapes_basic():
    image_shape = (300, 300)
    strides = [16, 32, 64]
    expected = [(19, 19), (10, 10), (5, 5)]
    actual = calculate_feature_map_shapes(image_shape, strides)
    assert actual == expected


def test_calculate_feature_map_shapes_invalid_inputs_raise():
    with pytest.raises(AssertionError, match=r"image_shape must be \(int, int\) pixels"):
        calculate_feature_map_shapes((300.5, 300), [16, 32])

    with pytest.raises(AssertionError, match="Image shape cannot be 0"):
        calculate_feature_map_shapes((300, 0), [16, 32])

    with pytest.raises(AssertionError, match="all strides must be positive ints"):
        calculate_feature_map_shapes((300, 300), [-16, 32])


def test_compute_scales_per_layer_explicit_and_include_extra():
    # Your signature requires s_min and s_max even when scales is explicit.
    scales = [0.2, 0.4, 0.6]
    out = compute_scales_per_layer(
        scales=scales,
        number_of_layers=3,
        s_min=0.2,
        s_max=0.9,
        include_extra=True,
    )

    assert len(out) == 3
    assert all(len(layer_scales) == 2 for layer_scales in out)

    tf.debugging.assert_near(out[0][0], tf.constant(0.2, tf.float32), atol=1e-6)
    tf.debugging.assert_near(out[1][0], tf.constant(0.4, tf.float32), atol=1e-6)
    tf.debugging.assert_near(out[2][0], tf.constant(0.6, tf.float32), atol=1e-6)


def test_standardize_aspect_ratios_inserts_one_and_pads_to_layers():
    aspect_ratios = [
        [2.0, 0.5],
        [1.0, 2.0],
    ]
    out = standardize_aspect_ratios(aspect_ratios=aspect_ratios, number_of_layers=3)

    assert len(out) == 3
    assert 1.0 in out[0]
    assert 1.0 in out[1]
    assert 1.0 in out[2]


def test_make_grid_centers_shape_and_range():
    # make_grid_centers is a tf.function with a fixed input_signature (float32 tensors),
    # so pass tensors (not tuples) and don't attempt to pass dtype.
    cx, cy = make_grid_centers(
        tf.constant([2.0, 3.0], tf.float32),
        tf.constant([300.0, 300.0], tf.float32),
    )

    tf.debugging.assert_rank(cx, 1)
    tf.debugging.assert_rank(cy, 1)
    tf.debugging.assert_equal(tf.size(cx), tf.constant(6, tf.int32))
    tf.debugging.assert_equal(tf.size(cy), tf.constant(6, tf.int32))

    tf.debugging.assert_greater_equal(tf.reduce_min(cx), 0.0)
    tf.debugging.assert_greater_equal(tf.reduce_min(cy), 0.0)
    tf.debugging.assert_less_equal(tf.reduce_max(cx), 1.0)
    tf.debugging.assert_less_equal(tf.reduce_max(cy), 1.0)


def test_anchors_per_cell_basic_shape():
    scales_in_layer = [0.2, 0.3]
    ratios_in_layer = [1.0, 2.0, 0.5]
    wh = anchors_per_cell(scales_in_layer, ratios_in_layer, dtype=tf.float32)

    tf.debugging.assert_rank(wh, 2)
    tf.debugging.assert_equal(tf.shape(wh)[1], 2)
    tf.debugging.assert_all_finite(wh, "anchors_per_cell produced non-finite values")


def test_tile_layer_anchors_output_shape():
    cx = tf.constant([0.25, 0.75], tf.float32)
    cy = tf.constant([0.25, 0.75], tf.float32)
    wh_cell = tf.constant([[0.2, 0.2], [0.1, 0.3]], tf.float32)

    priors = tile_layer_anchors(cx, cy, wh_cell, dtype=tf.float32)
    tf.debugging.assert_equal(tf.shape(priors), tf.constant([4, 4], tf.int32))
    tf.debugging.assert_all_finite(priors, "tile_layer_anchors produced non-finite values")


def test_build_layer_priors_clips_to_unit_interval():
    priors = build_layer_priors(
        feature_map_shape=(1, 1),
        image_size=(10, 10),
        scales_in_layer=[2.0],
        ratios_in_layer=[1.0],
        dtype=tf.float32,
        clip=True,
    )

    tf.debugging.assert_greater_equal(tf.reduce_min(priors), 0.0)
    tf.debugging.assert_less_equal(tf.reduce_max(priors), 1.0)


def test_concatenate_priors_empty_raises():
    with pytest.raises(ValueError, match="must be a non-empty list"):
        concatenate_priors([])


def test_concatenate_priors_clips_to_unit_interval():
    layer = tf.constant(
        [
            [-0.1, 0.5, 0.2, 0.2],
            [1.1,  0.5, 0.2, 0.2],
        ],
        dtype=tf.float32,
    )
    priors = concatenate_priors([layer], clip=True, dtype=tf.float32)

    tf.debugging.assert_greater_equal(tf.reduce_min(priors), 0.0)
    tf.debugging.assert_less_equal(tf.reduce_max(priors), 1.0)


def test_build_priors_requires_strides_when_feature_map_shapes_none():
    with pytest.raises(TypeError):
        build_priors(
            image_size=(300, 300),
            strides=None,
            feature_map_shapes=None,
            scales=[0.2, 0.4, 0.6],
            s_min=0.2,
            s_max=0.9,
            aspect_ratios=[[1.0]],
            include_extra=False,
            clip=True,
            dtype=tf.float32,
            return_meta=False,
        )


def test_build_priors_and_meta_consistency_smoke():
    priors, meta = build_priors(
        image_size=(300, 300),
        strides=[16, 32],
        feature_map_shapes=None,
        scales=None,
        s_min=0.2,
        s_max=0.9,
        include_extra=True,
        aspect_ratios=[[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]],
        clip=True,
        dtype=tf.float32,
        return_meta=True,
    )

    tf.debugging.assert_rank(priors, 2)
    tf.debugging.assert_equal(tf.shape(priors)[1], 4)
    tf.debugging.assert_all_finite(priors, "priors contain non-finite values")

    assert isinstance(meta, dict)
    assert "total_number_of_anchors" in meta
    tf.debugging.assert_equal(meta["total_number_of_anchors"], tf.shape(priors)[0])


def test_build_priors_batched_repeats_priors_across_batch():
    priors, _ = build_priors(
        image_size=(300, 300),
        strides=[16, 32],
        feature_map_shapes=None,
        scales=None,
        s_min=0.2,
        s_max=0.9,
        include_extra=True,
        aspect_ratios=[[1.0, 2.0, 0.5], [1.0, 2.0, 0.5]],
        clip=True,
        dtype=tf.float32,
        return_meta=True,
    )

    batched = build_priors_batched(priors, batch_size=3)
    tf.debugging.assert_equal(tf.shape(batched)[0], 3)
    tf.debugging.assert_equal(tf.shape(batched)[1], tf.shape(priors)[0])
    tf.debugging.assert_near(batched[0], priors, atol=1e-6)
    tf.debugging.assert_near(batched[1], priors, atol=1e-6)
    tf.debugging.assert_near(batched[2], priors, atol=1e-6)
