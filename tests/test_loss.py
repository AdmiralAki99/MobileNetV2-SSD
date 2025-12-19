import tensorflow as tf
import pytest

from mobilenetv2ssd.models.ssd.ops.loss_ops_tf import *

def test_smooth_l1_loss_sum():
    
    predicted = tf.constant([[1.0, 2.0, 0.0, -2.0]], dtype=tf.float32)
    target    = tf.constant([[1.1, 1.0, 0.0, -3.0]], dtype=tf.float32)
    beta = 1.0
    
    loss = smooth_l1_loss(predicted, target, beta=beta, reduction="sum")
    
    expected = tf.constant(1.005, dtype=tf.float32)
    
    tf.debugging.assert_near(loss, expected, atol=1e-6, message="Smooth L1 loss mismatch for mixed regions")
    
def test_smooth_l1_loss_mean():
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, 1.0]], dtype=tf.float32)
    
    target = tf.zeros_like(predicted)
    beta = 1.0
    
    loss = smooth_l1_loss(predicted, target, beta=beta, reduction="mean")
    
    tf.debugging.assert_near(loss, tf.constant(0.5, dtype=tf.float32), message="mean reduction incorrect")
    
def test_smooth_l1_loss_max():
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, 1.0]], dtype=tf.float32)
    
    target = tf.zeros_like(predicted)
    beta = 1.0
    
    loss = smooth_l1_loss(predicted, target, beta=beta, reduction="max")
    
    tf.debugging.assert_near(loss,  tf.constant(1.0, dtype=tf.float32), message="max reduction incorrect")
    
def test_smooth_l1_none():
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, 1.0]], dtype=tf.float32)
    
    target = tf.zeros_like(predicted)
    beta = 1.0
    
    loss = smooth_l1_loss(predicted, target, beta=beta, reduction="none")
    
    tf.debugging.assert_near(loss, tf.constant([0.0, 1.0], dtype=tf.float32), message="no reduction output incorrect")
    
def test_l1_loss_single_sum():
    # predicted - target = [1, 2, 3, 4]
    predicted = tf.constant([[1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)
    target    = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)

    loss = l1_loss(predicted, target, reduction="sum")

    # |1| + |2| + |3| + |4| = 10
    expected = tf.constant(10.0, dtype=tf.float32)
    tf.debugging.assert_near(loss, expected, atol=1e-6, message="L1 loss (sum) mismatch for single example")
    
def test_l1_loss_mean():
    # Two rows:
    # row 0: diff = [0, 0]   → row L1 = 0
    # row 1: diff = [1, -2]  → row L1 = |1| + | -2 | = 3
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, -2.0]], dtype=tf.float32)
    target    = tf.constant([[0.0, 0.0],
                             [0.0,  0.0]], dtype=tf.float32)
    
    loss = l1_loss(predicted, target, reduction="mean")
    
    tf.debugging.assert_near(
        loss,
        tf.constant(1.5, dtype=tf.float32),  # (0 + 3) / 2
        message="L1 loss mean reduction incorrect",
    )
    
def test_l1_loss_max():
    # Two rows:
    # row 0: diff = [0, 0]   → row L1 = 0
    # row 1: diff = [1, -2]  → row L1 = |1| + | -2 | = 3
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, -2.0]], dtype=tf.float32)
    target    = tf.constant([[0.0, 0.0],
                             [0.0,  0.0]], dtype=tf.float32)
    
    loss = l1_loss(predicted, target, reduction="max")
    
    tf.debugging.assert_near(
        loss,
        tf.constant(3.0, dtype=tf.float32),
        message="L1 loss max reduction incorrect",
    )
    
def test_l1_loss_none():
    # Two rows:
    # row 0: diff = [0, 0]   → row L1 = 0
    # row 1: diff = [1, -2]  → row L1 = |1| + | -2 | = 3
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, -2.0]], dtype=tf.float32)
    target    = tf.constant([[0.0, 0.0],
                             [0.0,  0.0]], dtype=tf.float32)
    
    loss = l1_loss(predicted, target, reduction="none")
    
    tf.debugging.assert_near(
        loss,
        tf.constant([0.0, 3.0], dtype=tf.float32),
        message="L1 loss no-reduction output incorrect",
    )
    
def test_l2_loss_single_example_sum():
    # predicted - target = [1, 2, 3, 4]
    predicted = tf.constant([[1.0, 2.0, 3.0, 4.0]], dtype=tf.float32)
    target    = tf.constant([[0.0, 0.0, 0.0, 0.0]], dtype=tf.float32)

    loss = l2_loss(predicted, target, reduction="sum")

    # L2 per coord: 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
    expected = tf.constant(30.0, dtype=tf.float32)

    tf.debugging.assert_near(
        loss,
        expected,
        atol=1e-6,
        message="L2 loss (sum) mismatch for single example",
    )

def test_l2_loss_sum():
    
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, -2.0]], dtype=tf.float32)
    target    = tf.constant([[0.0, 0.0],
                             [0.0,  0.0]], dtype=tf.float32)

    loss = l2_loss(predicted, target, reduction="sum")

    tf.debugging.assert_near(
        loss,
        tf.constant(5.0, dtype=tf.float32),
        message="L2 loss sum reduction incorrect",
    )
    
def test_l2_loss_mean():
    
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, -2.0]], dtype=tf.float32)
    target    = tf.constant([[0.0, 0.0],
                             [0.0,  0.0]], dtype=tf.float32)

    loss = l2_loss(predicted, target, reduction="mean")

    tf.debugging.assert_near(
        loss,
        tf.constant(2.5, dtype=tf.float32),  # (0 + 5) / 2
        message="L2 loss mean reduction incorrect",
    )
    
def test_l2_loss_max():
    
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, -2.0]], dtype=tf.float32)
    target    = tf.constant([[0.0, 0.0],
                             [0.0,  0.0]], dtype=tf.float32)

    loss = l2_loss(predicted, target, reduction="max")

    tf.debugging.assert_near(
        loss,
        tf.constant(5.0, dtype=tf.float32),
        message="L2 loss max reduction incorrect",
    )
    
def test_l2_loss_none():
    predicted = tf.constant([[0.0, 0.0],
                             [1.0, -2.0]], dtype=tf.float32)
    target    = tf.constant([[0.0, 0.0],
                             [0.0,  0.0]], dtype=tf.float32)

    loss = l2_loss(predicted, target, reduction="none")

    tf.debugging.assert_near(
        loss,
        tf.constant([0.0, 5.0], dtype=tf.float32),
        message="L2 loss none reduction incorrect",
    )
      
def test_softmax_cross_entropy_none_single_example():
    # Logits for one sample: [0, 1, 2], label = 2
    logits = tf.constant([[0.0, 1.0, 2.0]], dtype=tf.float32)
    labels = tf.constant([2], dtype=tf.int32)

    loss = softmax_cross_entropy_loss(logits, labels, reduction="none")

    # Manually computed:
    # exps = [e^0, e^1, e^2]
    # p_2 = e^2 / (e^0 + e^1 + e^2)
    # loss = -log(p_2) ≈ 0.40760596
    expected = tf.constant([0.40760598], dtype=tf.float32)

    tf.debugging.assert_near(
        loss,
        expected,
        atol=1e-5,
        message="Softmax CE (none) mismatch for single example",
    )
    
def test_softmax_cross_entropy_none():
    # Two identical logits, different labels
    # Sample 0: label 2  → loss ≈ 0.40760596
    # Sample 1: label 1  → loss ≈ 1.40760596
    logits = tf.constant(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=tf.float32,
    )
    labels = tf.constant([2, 1], dtype=tf.int32)

    loss = softmax_cross_entropy_loss(logits, labels, reduction="none")

    expected = tf.constant(
        [0.40760598, 1.40760596],
        dtype=tf.float32,
    )

    # No reduction
    tf.debugging.assert_near(
        loss,
        expected,
        atol=1e-5,
        message="Softmax CE none reduction incorrect",
    )
    
def test_softmax_cross_entropy_sum():
    # Two identical logits, different labels
    # Sample 0: label 2  → loss ≈ 0.40760596
    # Sample 1: label 1  → loss ≈ 1.40760596
    logits = tf.constant(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=tf.float32,
    )
    labels = tf.constant([2, 1], dtype=tf.int32)

    loss  = softmax_cross_entropy_loss(logits, labels, reduction="sum")

    expected  = tf.constant(1.8152119, dtype=tf.float32)

    # No reduction
    tf.debugging.assert_near(
        loss,
        expected,
        atol=1e-5,
        message="Softmax CE sum reduction incorrect",
    )
    
def test_softmax_cross_entropy_mean():
    # Two identical logits, different labels
    # Sample 0: label 2  → loss ≈ 0.40760596
    # Sample 1: label 1  → loss ≈ 1.40760596
    logits = tf.constant(
        [
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=tf.float32,
    )
    labels = tf.constant([2, 1], dtype=tf.int32)

    loss = softmax_cross_entropy_loss(logits, labels, reduction="mean")

    expected = tf.constant(0.90760595, dtype=tf.float32)

    # No reduction
    tf.debugging.assert_near(
        loss,
        expected,
        atol=1e-5,
        message="Softmax CE mean reduction incorrect",
    )

def test_multibox_loss_basic():
    # One batch, two anchors, three classes
    B, N, C = 1, 2, 3

    # Offsets:
    # Anchor 0: positive, small nonzero offsets
    # Anchor 1: negative, zeros (won't be used in localization loss)
    predicted_offsets = tf.constant(
        [
            [
                [0.1, -0.1, 0.2, -0.2],  # positive
                [0.0,  0.0,  0.0,  0.0],  # negative
            ]
        ],
        dtype=tf.float32,
    )
    target_offsets = tf.zeros_like(predicted_offsets)

    # Logits: all zeros → uniform probabilities over 3 classes
    # For any label, CE = -log(1/3) = log(3)
    predicted_logits = tf.zeros((B, N, C), dtype=tf.float32)

    # Labels: anchor 0 = class 1, anchor 1 = background (class 0)
    target_labels = tf.constant([[1, 0]], dtype=tf.int32)

    # Masks: anchor 0 positive, anchor 1 negative
    positive_mask = tf.constant([[True, False]], dtype=tf.bool)
    negative_mask = tf.constant([[False, True]], dtype=tf.bool)

    localization_weight = 1.0
    classification_weight = 1.0
    beta = 1.0

    out = multibox_loss(
        predicted_offsets=predicted_offsets,
        predicted_logits=predicted_logits,
        target_offsets=target_offsets,
        target_labels=target_labels,
        positive_mask=positive_mask,
        negative_mask=negative_mask,
        localization_weight=localization_weight,
        classification_weight=classification_weight,
        beta=beta,
        cls_loss_type="softmax_ce",
        loc_loss_type="smooth_l1",
        normalize_denom="num_pos",  # default
        reduction="sum",            # default
    )

    # --- Expected localization loss (smooth L1 over positive anchor only) ---
    # d = [0.1, -0.1, 0.2, -0.2], |d| < beta=1 → all in quadratic region
    # per coord: 0.5 * d^2
    d = tf.constant([0.1, -0.1, 0.2, -0.2], dtype=tf.float32)
    expected_loc_raw = 0.5 * tf.reduce_sum(d * d)  # = 0.05
    # normalize_denom="num_pos" with 1 positive → loc_loss = raw / 1
    expected_loc_loss = expected_loc_raw

    # --- Expected classification loss (softmax CE over pos+neg) ---
    # logits are all zeros → p = [1/3, 1/3, 1/3]
    # loss per sample = -log(1/3) = log(3)
    # two anchors selected (1 pos + 1 neg) → raw = 2 * log(3)
    expected_cls_raw = 2.0 * tf.math.log(3.0)
    # normalize_denom="num_pos" → divide by 1 positive
    expected_cls_loss = expected_cls_raw

    expected_total = expected_loc_loss + expected_cls_loss
    expected_num_pos = tf.constant(1, dtype=tf.int32)
    expected_num_neg = tf.constant(1, dtype=tf.int32)

    # --- Assertions ---
    tf.debugging.assert_near(
        out["loc_loss"],
        expected_loc_loss,
        atol=1e-6,
        message="Localization loss mismatch",
    )
    tf.debugging.assert_near(
        out["cls_loss"],
        expected_cls_loss,
        atol=1e-6,
        message="Classification loss mismatch",
    )
    tf.debugging.assert_near(
        out["total_loss"],
        expected_total,
        atol=1e-6,
        message="Total multibox loss mismatch",
    )
    tf.debugging.assert_equal(
        out["num_pos"],
        expected_num_pos,
        message="Number of positives mismatch",
    )
    tf.debugging.assert_equal(
        out["num_negative"],
        expected_num_neg,
        message="Number of negatives mismatch",
    )
    
def test_multibox_loss_no_positives_has_zero_loc_and_safe_normalization():
    # No positives: loc_loss should be 0, cls_loss should be computed on negatives only,
    B, N, C = 1, 3, 3

    predicted_offsets = tf.constant([[[0.1, 0.2, -0.1, 0.0],
                                     [0.0, 0.0,  0.0, 0.0],
                                     [0.3, -0.2, 0.2, -0.1]]], dtype=tf.float32)
    target_offsets    = tf.zeros_like(predicted_offsets)

    # Simple logits
    predicted_logits = tf.zeros((B, N, C), dtype=tf.float32)

    # Labels (background = 0)
    target_labels = tf.zeros((B, N), dtype=tf.int32)

    positive_mask = tf.zeros((B, N), dtype=tf.bool)
    negative_mask = tf.constant([[True, True, False]], dtype=tf.bool)  # select 2 negatives

    out = multibox_loss(
        predicted_offsets=predicted_offsets,
        predicted_logits=predicted_logits,
        target_offsets=target_offsets,
        target_labels=target_labels,
        positive_mask=positive_mask,
        negative_mask=negative_mask,
        localization_weight=1.0,
        classification_weight=1.0,
        beta=1.0,
        cls_loss_type="softmax_ce",
        loc_loss_type="smooth_l1",
        normalize_denom="num_pos",
        reduction="sum",
    )

    tf.debugging.assert_near(out["loc_loss"], 0.0, atol=1e-6, message="loc_loss should be 0 when there are no positives")

    expected_cls = 2.0 * tf.math.log(3.0)
    tf.debugging.assert_near(out["cls_loss"], expected_cls, atol=1e-6, message="cls_loss mismatch for no-positives case")

    # By implementation, num_pos is the count; should be 0 here
    tf.debugging.assert_equal(out["num_pos"], tf.constant(0, dtype=tf.int32), message="num_pos should be 0 for no-positives case")
    tf.debugging.assert_equal(out["num_negative"], tf.constant(2, dtype=tf.int32), message="num_negative mismatch for no-positives case")


def test_multibox_loss_no_negatives_is_still_well_defined():
    # No negatives: classification should still include positives (classification_mask = pos OR neg).
    B, N, C = 1, 2, 3

    predicted_offsets = tf.constant([[[0.1, -0.1, 0.0, 0.2],
                                     [0.2,  0.1, -0.1, 0.0]]], dtype=tf.float32)
    target_offsets = tf.zeros_like(predicted_offsets)

    predicted_logits = tf.zeros((B, N, C), dtype=tf.float32)  # uniform -> CE = log(3)
    target_labels = tf.constant([[1, 2]], dtype=tf.int32)     # both are positives classes

    positive_mask = tf.constant([[True, True]], dtype=tf.bool)
    negative_mask = tf.zeros((B, N), dtype=tf.bool)

    out = multibox_loss(
        predicted_offsets=predicted_offsets,
        predicted_logits=predicted_logits,
        target_offsets=target_offsets,
        target_labels=target_labels,
        positive_mask=positive_mask,
        negative_mask=negative_mask,
        localization_weight=1.0,
        classification_weight=1.0,
        beta=1.0,
        cls_loss_type="softmax_ce",
        loc_loss_type="smooth_l1",
        normalize_denom="num_pos",
        reduction="sum",
    )

    expected_cls = tf.math.log(3.0)
    tf.debugging.assert_near(out["cls_loss"], expected_cls, atol=1e-6, message="cls_loss mismatch for no-negatives case")

    tf.debugging.assert_greater_equal(out["loc_loss"], 0.0, message="loc_loss should be non-negative")
    tf.debugging.assert_all_finite(out["loc_loss"], "loc_loss should be finite")
    tf.debugging.assert_all_finite(out["total_loss"], "total_loss should be finite")

    tf.debugging.assert_equal(out["num_pos"], tf.constant(2, dtype=tf.int32), message="num_pos mismatch for no-negatives case")
    tf.debugging.assert_equal(out["num_negative"], tf.constant(0, dtype=tf.int32), message="num_negative should be 0 for no-negatives case")

    