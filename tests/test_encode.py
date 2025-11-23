import tensorflow as tf

from mobilenetv2ssd.models.ssd.ops.encode_ops_tf import *

def test_encode():
    priors_cxcywh = tf.constant([[0.5, 0.5, 0.2, 0.2]], dtype = tf.float32)
    gt_xyxy = tf.constant([[0.44, 0.4, 0.64, 0.6]], dtype = tf.float32)
    variance = (0.1,0.2)
    
    expected_encoding = tf.constant([ 1.9999981e+00,  0.0000000e+00, -2.9802322e-07,  5.9604639e-07], dtype=tf.float32)
    
    matched_encoding = encode_boxes_core(gt_xyxy,priors_cxcywh,variance)
    
    tf.debugging.assert_near(expected_encoding, matched_encoding)
    
def test_encode_batched():
    matched_gt_xyxy_batch = tf.constant(
    [
        # Image 0: (N=3 priors)
        [
            [0.1, 0.1, 0.3, 0.3],  # prior 0 matched to GT box A
            [0.6, 0.6, 0.9, 0.9],  # prior 1 matched to GT box B
            [0.0, 0.0, 0.0, 0.0],  # prior 2 background (padded)
        ],
        # Image 1:
        [
            [0.0, 0.0, 0.0, 0.0],  # prior 0 background
            [0.6, 0.6, 0.9, 0.9],  # prior 1 matched to GT box B
            [0.6, 0.6, 0.9, 0.9],  # prior 2 also matched to GT box B
        ],
    ],
    dtype=tf.float32)
    priors = tf.constant(
        [
            [0.2, 0.2, 0.2, 0.2],  # prior 0
            [0.7, 0.7, 0.3, 0.3],  # prior 1
            [0.5, 0.5, 0.4, 0.4],  # prior 2
        ],
    dtype=tf.float32)
    
    matched_encoding = encode_boxes_batch(matched_gt_xyxy_batch,priors,(0.1, 0.2)) 
    
    expected_encoding = tf.constant(
        [
            [
                [ 0.0000000e+00,  0.0000000e+00,  5.9604639e-07,  5.9604639e-07],
                [ 1.6666670e+00,  1.6666670e+00, -8.9406973e-07, -8.9406973e-07],
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00]
            ],

            [
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                [ 1.6666670e+00,  1.6666670e+00, -8.9406973e-07, -8.9406973e-07],
                [ 6.2500000e+00,  6.2500000e+00, -1.4384111e+00, -1.4384111e+00]
            ]
        ],
    )
    
    tf.debugging.assert_near(expected_encoding, matched_encoding)