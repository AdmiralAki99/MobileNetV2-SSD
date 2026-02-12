import tensorflow as tf
import numpy as np
import math

from mobilenetv2ssd.models.ssd.ops.encode_ops_tf import *

import pytest
pytestmark = pytest.mark.unit

def _xyxy_from_cxcywh(cxcywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = cxcywh
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def _expected_encode(gt_xyxy: np.ndarray, prior_cxcywh: np.ndarray, variance=(0.1, 0.2)) -> np.ndarray:
    """Compute expected SSD-style encoding for a single box."""
    px, py, pw, ph = prior_cxcywh
    x1, y1, x2, y2 = gt_xyxy

    gcx = (x1 + x2) / 2.0
    gcy = (y1 + y2) / 2.0
    gw = (x2 - x1)
    gh = (y2 - y1)

    v0, v1 = variance
    tx = (gcx - px) / (pw * v0)
    ty = (gcy - py) / (ph * v0)
    tw = math.log(gw / pw) / v1
    th = math.log(gh / ph) / v1
    return np.array([tx, ty, tw, th], dtype=np.float32)


def test_encode_core_simple_values():
    priors_cxcywh = tf.constant([[0.5, 0.5, 0.2, 0.2]], dtype=tf.float32)
    gt_xyxy = tf.constant([[0.44, 0.40, 0.64, 0.60]], dtype=tf.float32)
    variance = (0.1, 0.2)

    # This GT has the same width/height as the prior, centered 0.04 to the right.
    # Expected: tx=2, ty=0, tw=0, th=0.
    expected = tf.constant([2.0, 0.0, 0.0, 0.0], dtype=tf.float32)

    encoded = encode_boxes_core(gt_xyxy, priors_cxcywh, variance)

    tf.debugging.assert_near(encoded, expected, atol=1e-5, rtol=1e-5)


def test_encode_core_scale_terms_nonzero():
    priors_cxcywh = tf.constant([[0.5, 0.5, 0.2, 0.2]], dtype=tf.float32)

    # Make GT box twice as wide/tall as the prior (same center), so tw/th are nonzero.
    gt_xyxy_np = _xyxy_from_cxcywh(np.array([0.5, 0.5, 0.4, 0.4], dtype=np.float32))
    gt_xyxy = tf.constant([gt_xyxy_np], dtype=tf.float32)

    variance = (0.1, 0.2)
    expected_np = _expected_encode(gt_xyxy_np, np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32), variance)
    expected = tf.constant(expected_np, dtype=tf.float32)

    encoded = encode_boxes_core(gt_xyxy, priors_cxcywh, variance)
    tf.debugging.assert_near(encoded, expected, atol=1e-5, rtol=1e-5)


def test_encode_batched_matches_core_for_each_row():
    variance = (0.1, 0.2)

    # Shared priors across the batch: [N, 4]
    priors_np = np.array(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.6, 0.6, 0.3, 0.3],
            [0.4, 0.4, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    # GT boxes: [B, N, 4]
    gt_np = np.zeros((2, 3, 4), dtype=np.float32)

    # image 0
    gt_np[0, 0] = _xyxy_from_cxcywh(priors_np[0])  # identical -> zeros
    gt_np[0, 1] = _xyxy_from_cxcywh(np.array([0.65, 0.65, 0.3, 0.3], dtype=np.float32))  # shift
    gt_np[0, 2] = _xyxy_from_cxcywh(np.array([0.4, 0.4, 0.2, 0.2], dtype=np.float32))   # bigger

    # image 1 (same priors, different GT)
    gt_np[1, 0] = _xyxy_from_cxcywh(priors_np[0])  # identical -> zeros
    gt_np[1, 1] = _xyxy_from_cxcywh(np.array([0.55, 0.55, 0.3, 0.3], dtype=np.float32))  # shift opposite
    gt_np[1, 2] = _xyxy_from_cxcywh(np.array([0.4, 0.4, 0.05, 0.05], dtype=np.float32))  # smaller

    priors = tf.constant(priors_np, dtype=tf.float32)      # [N,4]
    gt_xyxy = tf.constant(gt_np, dtype=tf.float32)         # [B,N,4]

    encoded_batched = encode_boxes_batch(gt_xyxy, priors, variance)

    # Compare each image's encoded result to the core call.
    encoded0 = encode_boxes_core(gt_xyxy[0], priors, variance)
    encoded1 = encode_boxes_core(gt_xyxy[1], priors, variance)

    tf.debugging.assert_near(encoded_batched[0], encoded0, atol=1e-5, rtol=1e-5)
    tf.debugging.assert_near(encoded_batched[1], encoded1, atol=1e-5, rtol=1e-5)
