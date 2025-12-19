import tensorflow as tf
import pytest

from mobilenetv2ssd.models.ssd.ops.match_ops_tf import *

import pytest
pytestmark = pytest.mark.unit

def test_match_priors():
    gt_boxes_xyxy = tf.constant(
        [
            [0.1, 0.1, 0.3, 0.3],  # A (class 3)
            [0.6, 0.6, 0.9, 0.9],  # B (class 2)
        ]
    ,tf.float32)
    gt_labels = tf.constant([3, 2], tf.int32)
    priors_cxcywh = tf.constant(
        [
            [0.2,  0.2,  0.20, 0.20],   # P0 -> IoU 1.00 with A
            [0.2,  0.2,  0.16, 0.16],   # P1 -> IoU 0.64 with A
            [0.2,  0.2,  0.12, 0.12],   # P2 -> IoU 0.36 with A  (ignore band)
            [0.75, 0.75, 0.30, 0.30],   # P3 -> IoU 1.00 with B
            [0.75, 0.75, 0.24, 0.24],   # P4 -> IoU 0.64 with B
            [0.05, 0.90, 0.10, 0.10],   # P5 -> IoU 0.00 (negative)
        ]
    , tf.float32)
    
    gt_valid_mask = tf.constant([True,True], dtype = tf.bool)
    
    positive_iou_thresh = 0.5
    negative_iou_thresh = 0.3
    
    matched_priors = match_priors(priors_cxcywh,gt_boxes_xyxy,gt_labels,gt_valid_mask,positive_iou_thresh,0.3,negative_iou_thresh,center_in_gt = True,allow_low_qual_matches = False,return_iou = True)
    
    expected_priors = tf.constant(
        [
            [0.1, 0.1, 0.3, 0.3],
            [0.1, 0.1, 0.3, 0.3],
            [0. , 0. , 0. , 0. ],
            [0.6, 0.6, 0.9, 0.9],
            [0.6, 0.6, 0.9, 0.9],
            [0. , 0. , 0. , 0. ]
        ]
        , tf.float32)
    expected_matched_gt_labels = tf.constant([3, 3, 0, 2, 2, 0], tf.int32)
    expected_pos_mask = tf.constant([ True,  True, False,  True,  True, False], tf.bool)
    expected_neg_mask = tf.constant([False, False, False, False, False,  True], tf.bool)
    expected_ignore_mask = tf.constant([False, False,  True, False, False, False], tf.bool)
    expected_matched_gt_idx = tf.constant([ 0,  0, -1,  1,  1, -1], tf.int32)
    expected_num_pos = tf.constant(4,tf.int32)
    expected_matched_iou = tf.constant([1.        , 0.63999987, 0.        , 1.        , 0.6400002 , 0.        ], tf.float32)
    
    expected_priors = {
        "matched_gt_xyxy" : expected_priors,
        "matched_gt_labels": expected_matched_gt_labels,
        "pos_mask": expected_pos_mask,
        "neg_mask": expected_neg_mask,
        "ignore_mask": expected_ignore_mask,
        "matched_gt_idx": expected_matched_gt_idx,
        "num_pos": expected_num_pos,
        "matched_iou" : expected_matched_iou
    }
    
    assert set(expected_priors.keys()) == set(matched_priors.keys()), "Keys are not the same"
    
    for key in expected_priors.keys():
        expected = expected_priors[key]
        actual = matched_priors[key]
        
        assert actual.shape == expected.shape, f"Shape mismatch for key '{key}': {actual.shape} vs {expected.shape}"
        assert actual.dtype == expected.dtype, f"Dtype mismatch for key '{key}': {actual.dtype} vs {expected.dtype}"
        
        if actual.dtype.is_floating:
            tf.debugging.assert_near(actual,expected, message = f"Float  mismatch for key '{key}'")
        else:
            tf.debugging.assert_equal(actual,expected,message = f"Value mismatch for key '{key}'")

def test_match_priors_simple():
    # One GT box (class 3)
    gt_boxes_xyxy = tf.constant([[0.1, 0.1, 0.3, 0.3]], dtype=tf.float32)
    gt_labels = tf.constant([3], dtype=tf.int32)
    gt_valid_mask = tf.constant([True], dtype = tf.bool)
    # Two priors:
    # P0 centered in GT with same size → IoU = 1.0 → positive
    # P1 far away from GT → IoU ~ 0 → negative
    priors_cxcywh = tf.constant(
        [
            [0.2, 0.2, 0.2, 0.2],  # P0 → positive
            [0.8, 0.8, 0.2, 0.2],  # P1 → negative
        ],
        dtype=tf.float32,
    )

    positive_iou_thresh = 0.5
    negative_iou_thresh = 0.3

    matched = match_priors(
        priors_cxcywh=priors_cxcywh,
        gt_boxes_xyxy=gt_boxes_xyxy,
        gt_labels=gt_labels,
        gt_valid_mask= gt_valid_mask,
        positive_iou_thresh=positive_iou_thresh,
        negative_iou_thresh=negative_iou_thresh,
        max_pos_per_gt=None,
        allow_low_qual_matches=False,
        center_in_gt=True,
        return_iou=False,
    )

    # --- Expected values ---

    # P0 should be matched to the only GT box; P1 should be background
    expected_matched_gt_xyxy = tf.constant(
        [
            [0.1, 0.1, 0.3, 0.3],  # P0 → GT
            [0.0, 0.0, 0.0, 0.0],  # P1 → background
        ],
        dtype=tf.float32,
    )
    expected_matched_labels = tf.constant([3, 0], dtype=tf.int32)
    expected_pos_mask = tf.constant([True, False], dtype=tf.bool)
    expected_neg_mask = tf.constant([False, True], dtype=tf.bool)
    expected_num_pos = tf.constant(1, dtype=tf.int32)

    # --- Assertions (simple & explicit) ---

    tf.debugging.assert_near(
        matched["matched_gt_xyxy"],
        expected_matched_gt_xyxy,
        message="matched_gt_xyxy mismatch",
    )
    tf.debugging.assert_equal(
        matched["matched_gt_labels"],
        expected_matched_labels,
        message="matched_gt_labels mismatch",
    )
    tf.debugging.assert_equal(
        matched["pos_mask"],
        expected_pos_mask,
        message="pos_mask mismatch",
    )
    tf.debugging.assert_equal(
        matched["neg_mask"],
        expected_neg_mask,
        message="neg_mask mismatch",
    )
    tf.debugging.assert_equal(
        matched["num_pos"],
        expected_num_pos,
        message="num_pos mismatch",
    )
     
def test_match_empty_gt_boxes():
    
    gt_boxes_xyxy = tf.constant([],tf.float32)
    gt_labels = tf.constant([3, 2], tf.int32)
    gt_valid_mask = tf.constant([False], dtype = tf.bool)
    priors_cxcywh = tf.constant(
        [
            [0.2,  0.2,  0.20, 0.20],   # P0 -> IoU 1.00 with A
            [0.2,  0.2,  0.16, 0.16],   # P1 -> IoU 0.64 with A
            [0.2,  0.2,  0.12, 0.12],   # P2 -> IoU 0.36 with A  (ignore band)
            [0.75, 0.75, 0.30, 0.30],   # P3 -> IoU 1.00 with B
            [0.75, 0.75, 0.24, 0.24],   # P4 -> IoU 0.64 with B
            [0.05, 0.90, 0.10, 0.10],   # P5 -> IoU 0.00 (negative)
        ]
    , tf.float32)
    
    positive_iou_thresh = 0.5
    negative_iou_thresh = 0.3
    
    N = tf.shape(priors_cxcywh)[0]
    expected_priors ={
        "matched_gt_xyxy": tf.zeros([N, 4], tf.float32),
        "matched_gt_labels":  tf.zeros([N], tf.int32),
        "pos_mask":        tf.zeros([N], tf.bool),
        "neg_mask":        tf.ones([N],  tf.bool),
        "ignore_mask":     tf.zeros([N], tf.bool),
        "matched_gt_idx":  -tf.ones([N], tf.int32),
        "matched_iou":     tf.zeros([N], tf.float32),
        "num_pos":         tf.zeros([], tf.int32),
    }
    
    matched_priors = match_priors(priors_cxcywh,gt_boxes_xyxy,gt_labels,gt_valid_mask,positive_iou_thresh,0.3,negative_iou_thresh,center_in_gt = True,allow_low_qual_matches = False,return_iou = True)
    
    assert set(expected_priors.keys()) == set(matched_priors.keys()), "Keys are not the same"
    
    for key in expected_priors.keys():
        expected = expected_priors[key]
        actual = matched_priors[key]
        
        assert actual.shape == expected.shape, f"Shape mismatch for key '{key}': {actual.shape} vs {expected.shape}"
        assert actual.dtype == expected.dtype, f"Dtype mismatch for key '{key}': {actual.dtype} vs {expected.dtype}"
        
        if actual.dtype.is_floating:
            tf.debugging.assert_near(actual,expected, message = f"Float  mismatch for key '{key}'")
        else:
            tf.debugging.assert_equal(actual,expected,message = f"Value mismatch for key '{key}'")
            
def test_hard_negative_mining():
    pos_mask = tf.constant([False,  True, False, False,  True, False, False, False])
    neg_mask = tf.constant([ True, False,  True,  True, False,  True,  True,  True])
    conf_loss = tf.constant([0.05,  1.20,  0.80,  0.02,  0.40,  1.50,  0.30,  0.90])
    
    mined_neg_mask, mined_neg_indices = hard_negative_mining(conf_loss,pos_mask,neg_mask,neg_ratio = 1.0,min_neg = None, max_neg = None)
    
    expected_mask = tf.constant([False, False, False, False, False,  True, False,  True], tf.bool)
    expected_num_indices = tf.constant(2,tf.int32)
    
    tf.debugging.assert_equal(mined_neg_mask,expected_mask,message = f"Value mismatch for mined negative mask")
    tf.debugging.assert_equal(expected_num_indices,tf.shape(mined_neg_indices)[0],message = f"Value mismatch for num of mined indices")