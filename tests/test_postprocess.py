import tensorflow as tf

from mobilenetv2ssd.models.ssd.ops.postprocess_tf import *

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
    top_k          = 50           # big enough; not really limiting here
    max_detections = 3
    use_sigmoid    = False        # we’re using softmax with background
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