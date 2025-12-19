import tensorflow as tf

from mobilenetv2ssd.models.ssd.ops.postprocess_tf import _decode_boxes, decode_and_nms
from mobilenetv2ssd.models.ssd.ops.box_ops_tf import cxcywh_toxyxy_core
def test_decode_boxes_zero_offsets_matches_prior_geometry():
    # If offsets are all zeros, decoded boxes should equal the prior boxes in xyxy form.
    pred_loc = tf.zeros((1, 3, 4), dtype=tf.float32)

    priors = tf.constant(
        [
            [0.25, 0.25, 0.2, 0.2],
            [0.75, 0.25, 0.4, 0.2],
            [0.50, 0.75, 0.2, 0.4],
        ],
        dtype=tf.float32,
    )

    variances = tf.constant([0.1, 0.2], dtype=tf.float32)

    decoded = _decode_boxes(predicted_offsets=pred_loc, priors=priors, variances=variances)

    expected = cxcywh_toxyxy_core(priors)[tf.newaxis, ...]  # [1, N, 4]

    tf.debugging.assert_equal(tf.shape(decoded), tf.shape(expected), message="Decoded shape mismatch")
    tf.debugging.assert_near(decoded, expected, atol=1e-6, message="Decoded boxes should match priors for zero offsets")
    
def test_decode_and_nms_softmax_classes():
    # In softmax mode (use_sigmoid=False), decode_and_nms shifts classes by +1
    # after combined_non_max_suppression, so classes should be >= 1 for valid detections.
    pred_loc = tf.zeros((1, 2, 4), dtype=tf.float32)

    priors = tf.constant(
        [
            [0.25, 0.25, 0.2, 0.2],
            [0.75, 0.75, 0.2, 0.2],
        ],
        dtype=tf.float32,
    )

    # 3-class logits (no explicit background in this setup)
    pred_logits = tf.constant(
        [
            [
                [0.1, 3.0, 0.0],  # anchor 0 -> class 1 is best
                [0.1, 0.0, 3.0],  # anchor 1 -> class 2 is best
            ]
        ],
        dtype=tf.float32,
    )

    variances = tf.constant([0.1, 0.2], dtype=tf.float32)

    nmsed_boxes, nmsed_scores, nmsed_classes, valid = decode_and_nms(
        predicted_offsets=pred_loc,
        predicted_logits=pred_logits,
        priors=priors,
        variances=variances,
        scores_thresh=0.05,
        iou_thresh=0.5,
        top_k=10,
        max_detections=10,
        image_meta=None,
        use_sigmoid=False,
    )

    tf.debugging.assert_equal(valid, tf.constant([2], tf.int32), message="Expected both anchors to be kept")
    # Only check class indexing convention, not exact ordering.
    kept_classes = nmsed_classes[0, : valid[0]]
    tf.debugging.assert_greater_equal(
        tf.reduce_min(kept_classes),
        tf.constant(1, tf.int32),
        message="Softmax classes should be 1-indexed (>= 1) after +1 shift",
    )

def test_decode_and_nms_pre_nms_top_k():
    # pre_nms_top_k should restrict how many candidate anchors are passed into NMS.
    B, N, C = 1, 5, 3
    pred_loc = tf.zeros((B, N, 4), dtype=tf.float32)

    priors = tf.constant(
        [
            [0.10, 0.10, 0.1, 0.1],
            [0.30, 0.10, 0.1, 0.1],
            [0.50, 0.10, 0.1, 0.1],
            [0.70, 0.10, 0.1, 0.1],
            [0.90, 0.10, 0.1, 0.1],
        ],
        dtype=tf.float32,
    )

    # Make anchor 1 and 4 have the highest softmax scores.
    # (We don't need exact scores; ordering is enough.)
    pred_logits = tf.constant(
        [
            [
                [0.0, 0.5, 0.0],   # low
                [0.0, 4.0, 0.0],   # very high
                [0.0, 0.8, 0.0],   # medium
                [0.0, 0.2, 0.0],   # low
                [0.0, 3.5, 0.0],   # high
            ]
        ],
        dtype=tf.float32,
    )

    variances = tf.constant([0.1, 0.2], dtype=tf.float32)

    pre_nms_top_k = 2
    nmsed_boxes, nmsed_scores, nmsed_classes, valid = decode_and_nms(
        predicted_offsets=pred_loc,
        predicted_logits=pred_logits,
        priors=priors,
        variances=variances,
        scores_thresh=0.8,
        iou_thresh=0.5,
        top_k=10,
        max_detections=10,
        image_meta=None,
        use_sigmoid=False,
        pre_nms_top_k=pre_nms_top_k,
    )

    tf.debugging.assert_equal(
        valid,
        tf.constant([pre_nms_top_k], tf.int32),
        message="valid_detections should match pre_nms_top_k when NMS doesn't suppress",
    )
    
def test_decode_and_nms_min_box_size_filters():
    # min_box_size should zero-out boxes/scores smaller than threshold before NMS.
    B, N, C = 1, 2, 3

    # Offsets are zero, so decoded boxes come directly from priors.
    pred_loc = tf.zeros((B, N, 4), dtype=tf.float32)

    # prior 0: small (0.01 x 0.01)  -> should be filtered
    # prior 1: large (0.20 x 0.20)  -> should remain
    priors = tf.constant(
        [
            [0.25, 0.25, 0.01, 0.01],
            [0.75, 0.75, 0.20, 0.20],
        ],
        dtype=tf.float32,
    )

    # Give both anchors high logits so filtering is the only reason one disappears.
    pred_logits = tf.constant(
        [
            [
                [0.0, 5.0, 0.0],  # would be high
                [0.0, 5.0, 0.0],  # high
            ]
        ],
        dtype=tf.float32,
    )

    variances = tf.constant([0.1, 0.2], dtype=tf.float32)

    nmsed_boxes, nmsed_scores, nmsed_classes, valid = decode_and_nms(
        predicted_offsets=pred_loc,
        predicted_logits=pred_logits,
        priors=priors,
        variances=variances,
        scores_thresh=0.8,
        iou_thresh=0.5,
        top_k=10,
        max_detections=10,
        image_meta=None,
        use_sigmoid=False,
        min_box_size=0.05,
    )

    tf.debugging.assert_equal(valid, tf.constant([1], tf.int32), message="Only the large box should remain after min_box_size filtering")

    # The remaining box should match prior 1 geometry (in normalized xyxy).
    expected_large_xyxy = cxcywh_toxyxy_core(priors[1:2])[tf.newaxis, ...]  # [1,1,4]
    tf.debugging.assert_near(
        nmsed_boxes[:, :1, :],
        expected_large_xyxy,
        atol=1e-6,
        message="Remaining box should be the large prior after filtering",
    )


def test_postprocess():
    pred_loc = tf.constant(
        [
            [  # batch 0
                [ 0.0,  0.0,  0.0,  0.0],   # anchor 0
                [ 0.2,  0.0,  0.0,  0.0],   # anchor 1
                [ 0.0,  0.0,  0.5,  0.0],   # anchor 2
                [ 0.0, -0.2,  0.0, -0.5],   # anchor 3
            ]
        ]
    )

    pred_logits = tf.constant(
        [
            [  # batch 0
                [0.1,  2.0,  0.0],   # anchor 0
                [0.0,  0.5,  3.0],   # anchor 1
                [0.5,  1.5, -1.0],   # anchor 2
                [0.2, -0.5,  0.0],   # anchor 3
            ]
        ]
    )

    priors = tf.constant(
        [
            [0.25, 0.25, 0.2, 0.2],  # anchor 0 (top-left)
            [0.75, 0.25, 0.2, 0.2],  # anchor 1 (top-right)
            [0.25, 0.75, 0.2, 0.2],  # anchor 2 (bottom-left)
            [0.75, 0.75, 0.2, 0.2],  # anchor 3 (bottom-right)
        ]
    )

    variances      = tf.constant([0.1, 0.2])
    score_thresh   = 0.3
    iou_thresh     = 0.5
    top_k          = 50 
    max_detections = 3
    use_sigmoid    = False 
    image_meta     = {'image_height': 300, 'image_width': 300} 
    
    expected_boxes = tf.constant(
        [
            [
                [ 45.     , 196.2    , 105.     , 256.2    ],
                [ 45.     ,  45.     , 105.     , 105.     ],
                [195.     ,  41.84487, 255.     , 108.15513]
            ]
        ]
    , dtype = tf.float32)
    
    expected_scores = tf.constant([[0.88349205, 0.77826834, 0.68967205]], dtype = tf.float32)
    
    expected_classes = tf.constant([[2, 1, 1]], dtype=tf.int32)
    
    expected_detections = tf.constant([3], dtype=tf.int32)
    
    nmsed_boxes,nmsed_scores, nmsed_classes, valid_detections = decode_and_nms(pred_loc, pred_logits, priors, variances,score_thresh,iou_thresh,top_k,max_detections,image_meta,use_sigmoid)
    
    tf.debugging.assert_near(
        expected_boxes,
        nmsed_boxes,
        atol=1e-6,
        message="NMSed Boxes are not the same",
    )
    
    tf.debugging.assert_near(
        expected_scores,
        nmsed_scores,
        atol=1e-6,
        message="NMSed Scores are not the same",
    )
    
    tf.debugging.assert_equal(
        expected_classes,
        nmsed_classes,
        message="NMSed Classes are not the same",
    )
    
    tf.debugging.assert_equal(
        expected_detections,
        valid_detections,
        message="NMSed Detections are not the same",
    )