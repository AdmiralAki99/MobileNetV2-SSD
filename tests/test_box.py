import pytest
import tensorflow as tf

from mobilenetv2ssd.models.ssd.ops.box_ops_tf import *

# Testing the core conversion functions
def test_xyxy_to_cxcywh_core_simple():
    box_xyxy = tf.constant([[0.0,0.0,2.0,2.0]], dtype=tf.float32)
    expected_boxes_cxcywh = tf.constant([[1.0,1.0,2.0,2.0]], dtype=tf.float32)
    
    boxes_cxcywh = xyxy_to_cxcywh_core(box_xyxy)
    
    tf.debugging.assert_near(boxes_cxcywh, expected_boxes_cxcywh)
    
def test_cxcywh_toxyxy_core_simple():
    box_cxcywh = tf.constant([[1.0,1.0,2.0,2.0]], dtype=tf.float32)
    expected_boxes_xyxy = tf.constant([[0.0,0.0,2.0,2.0]], dtype=tf.float32)
    
    boxes_xyxy = cxcywh_toxyxy_core(box_cxcywh)
    
    tf.debugging.assert_near(boxes_xyxy, expected_boxes_xyxy)
    
    
# Now checking the round trip conversion
def test_round_trip_conversion_xyxy():
    original_boxes_xyxy = tf.constant([[0.0,0.0,2.0,2.0]], dtype=tf.float32)
    
    boxes_cxcywh = xyxy_to_cxcywh_core(original_boxes_xyxy)
    expected_boxes_cxcywh = cxcywh_toxyxy_core(boxes_cxcywh)
    
    tf.debugging.assert_near(expected_boxes_cxcywh, original_boxes_xyxy)
    
def test_round_trip_conversion_cxcywh():
    original_boxes_cxcywh = tf.constant([[1.0,1.0,2.0,2.0]], dtype=tf.float32)
    
    boxes_xyxy = cxcywh_toxyxy_core(original_boxes_cxcywh)
    expected_boxes_xyxy = xyxy_to_cxcywh_core(boxes_xyxy)
    
    tf.debugging.assert_near(expected_boxes_xyxy, original_boxes_cxcywh)
    
# Checking for box with zero area
def test_zero_area_box_xyxy_to_cxcywh():
    box_xyxy = tf.constant([[1.0,1.0,1.0,1.0]], dtype=tf.float32)
    expected_boxes_cxcywh = tf.constant([[1.0,1.0,0.0,0.0]], dtype=tf.float32)
    
    boxes_cxcywh = xyxy_to_cxcywh_core(box_xyxy)
    
    tf.debugging.assert_near(boxes_cxcywh, expected_boxes_cxcywh)
    
# Checking for empty area box
def test_empty_area_box_xyxy_to_cxcywh():
    box_xyxy = tf.constant([[2.0,2.0,1.0,1.0]], dtype=tf.float32)
    expected_boxes_cxcywh = tf.constant([[1.5,1.5,-1.0,-1.0]], dtype=tf.float32)
    
    boxes_cxcywh = xyxy_to_cxcywh_core(box_xyxy)
    
    tf.debugging.assert_near(boxes_cxcywh, expected_boxes_cxcywh)
    
def test_xyxy_to_cxcywh_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = xyxy_to_cxcywh_core(bad_boxes)
        
def test_cxcywh_to_xyxy_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = cxcywh_toxyxy_core(bad_boxes)
    
# Checking the batched conversion functions
def test_batched_xyxy_to_cxcywh():
    xy_boxes_batched = tf.constant(
    [
        [
            [  80.0, 120.0, 160.0, 180.0],  
            [ 220.0, 220.0, 420.0, 420.0], 
            [ 440.0,   0.0, 560.0, 200.0],  
            [ 530.0, 470.0, 630.0, 530.0],  
            [ 100.0, 450.0, 500.0, 550.0],  
            [ -10.0, -10.0, 110.0, 110.0],  
            [ 580.0, 170.0, 620.0, 470.0],  
            [  50.0,  80.0, 350.0, 120.0],
        ],
        [
            [  80.0, 120.0, 160.0, 180.0],  
            [ 220.0, 220.0, 420.0, 420.0], 
            [ 440.0,   0.0, 560.0, 200.0],  
            [ 530.0, 470.0, 630.0, 530.0],  
            [ 100.0, 450.0, 500.0, 550.0],  
            [ -10.0, -10.0, 110.0, 110.0],  
            [ 580.0, 170.0, 620.0, 470.0],  
            [  50.0,  80.0, 350.0, 120.0],
        ]   
    ], dtype=tf.float32
    )
    
    center_boxes_batched = tf.constant(
        [
            [
                [120.0, 150.0,  80.0,  60.0],
                [320.0, 320.0, 200.0, 200.0],
                [500.0, 100.0, 120.0, 200.0],
                [580.0, 500.0, 100.0,  60.0],
                [300.0, 500.0, 400.0, 100.0],
                [ 50.0,  50.0, 120.0, 120.0],
                [600.0, 320.0,  40.0, 300.0],
                [200.0, 100.0, 300.0,  40.0],
            ],
            [
                [120.0, 150.0,  80.0,  60.0],
                [320.0, 320.0, 200.0, 200.0],
                [500.0, 100.0, 120.0, 200.0],
                [580.0, 500.0, 100.0,  60.0],
                [300.0, 500.0, 400.0, 100.0],
                [ 50.0,  50.0, 120.0, 120.0],
                [600.0, 320.0,  40.0, 300.0],
                [200.0, 100.0, 300.0,  40.0],
            ]
    ]
    , dtype=tf.float32
    )
    
    converted_boxes = xyxy_to_cxcywh_batched(xy_boxes_batched)
    
    tf.debugging.assert_near(converted_boxes, center_boxes_batched)
    
    
def test_batched_cxcywh_to_xyxy():
    center_boxes_batched = tf.constant(
        [
            [
                [120.0, 150.0,  80.0,  60.0],
                [320.0, 320.0, 200.0, 200.0],
                [500.0, 100.0, 120.0, 200.0],
                [580.0, 500.0, 100.0,  60.0],
                [300.0, 500.0, 400.0, 100.0],
                [ 50.0,  50.0, 120.0, 120.0],
                [600.0, 320.0,  40.0, 300.0],
                [200.0, 100.0, 300.0,  40.0],
            ],
            [
                [120.0, 150.0,  80.0,  60.0],
                [320.0, 320.0, 200.0, 200.0],
                [500.0, 100.0, 120.0, 200.0],
                [580.0, 500.0, 100.0,  60.0],
                [300.0, 500.0, 400.0, 100.0],
                [ 50.0,  50.0, 120.0, 120.0],
                [600.0, 320.0,  40.0, 300.0],
                [200.0, 100.0, 300.0,  40.0],
            ]
    ]
    , dtype=tf.float32
    )
    
    xy_boxes_batched = tf.constant(
    [
        [
            [  80.0, 120.0, 160.0, 180.0],  
            [ 220.0, 220.0, 420.0, 420.0], 
            [ 440.0,   0.0, 560.0, 200.0],  
            [ 530.0, 470.0, 630.0, 530.0],  
            [ 100.0, 450.0, 500.0, 550.0],  
            [ -10.0, -10.0, 110.0, 110.0],  
            [ 580.0, 170.0, 620.0, 470.0],  
            [  50.0,  80.0, 350.0, 120.0],
        ],
        [
            [  80.0, 120.0, 160.0, 180.0],  
            [ 220.0, 220.0, 420.0, 420.0], 
            [ 440.0,   0.0, 560.0, 200.0],  
            [ 530.0, 470.0, 630.0, 530.0],  
            [ 100.0, 450.0, 500.0, 550.0],  
            [ -10.0, -10.0, 110.0, 110.0],  
            [ 580.0, 170.0, 620.0, 470.0],  
            [  50.0,  80.0, 350.0, 120.0],
        ]
    ], dtype=tf.float32
    )
    
    converted_boxes = cxcywh_toxyxy_batched(center_boxes_batched)
    
    tf.debugging.assert_near(converted_boxes, xy_boxes_batched)
    
# Testing the round trip for batched conversions
def test_batched_round_trip_xyxy():
    original_boxes_batched = tf.constant(
    [
        [
            [  80.0, 120.0, 160.0, 180.0],  
            [ 220.0, 220.0, 420.0, 420.0], 
            [ 440.0,   0.0, 560.0, 200.0],  
            [ 530.0, 470.0, 630.0, 530.0],  
            [ 100.0, 450.0, 500.0, 550.0],  
            [ -10.0, -10.0, 110.0, 110.0],  
            [ 580.0, 170.0, 620.0, 470.0],  
            [  50.0,  80.0, 350.0, 120.0],
        ],
        [
            [  80.0, 120.0, 160.0, 180.0],  
            [ 220.0, 220.0, 420.0, 420.0], 
            [ 440.0,   0.0, 560.0, 200.0],  
            [ 530.0, 470.0, 630.0, 530.0],  
            [ 100.0, 450.0, 500.0, 550.0],  
            [ -10.0, -10.0, 110.0, 110.0],  
            [ 580.0, 170.0, 620.0, 470.0],  
            [  50.0,  80.0, 350.0, 120.0],
        ]   
    ], dtype=tf.float32
    )
    
    converted_to_cxcywh = xyxy_to_cxcywh_batched(original_boxes_batched)
    round_trip_boxes = cxcywh_toxyxy_batched(converted_to_cxcywh)
    
    tf.debugging.assert_near(round_trip_boxes, original_boxes_batched)
    
    
def test_batched_round_trip_cxcywh():
    original_boxes_batched = tf.constant(
        [
            [
                [120.0, 150.0,  80.0,  60.0],
                [320.0, 320.0, 200.0, 200.0],
                [500.0, 100.0, 120.0, 200.0],
                [580.0, 500.0, 100.0,  60.0],
                [300.0, 500.0, 400.0, 100.0],
                [ 50.0,  50.0, 120.0, 120.0],
                [600.0, 320.0,  40.0, 300.0],
                [200.0, 100.0, 300.0,  40.0],
            ],
            [
                [120.0, 150.0,  80.0,  60.0],
                [320.0, 320.0, 200.0, 200.0],
                [500.0, 100.0, 120.0, 200.0],
                [580.0, 500.0, 100.0,  60.0],
                [300.0, 500.0, 400.0, 100.0],
                [ 50.0,  50.0, 120.0, 120.0],
                [600.0, 320.0,  40.0, 300.0],
                [200.0, 100.0, 300.0,  40.0],
            ]
    ]
    , dtype=tf.float32
    )
    
    converted_to_xyxy = cxcywh_toxyxy_batched(original_boxes_batched)
    round_trip_boxes = xyxy_to_cxcywh_batched(converted_to_xyxy)
    
    tf.debugging.assert_near(round_trip_boxes, original_boxes_batched)
    
    
# Checking for zero area boxes in batched conversions
def test_batched_zero_area_box_xyxy_to_cxcywh():
    box_xyxy_batched = tf.constant(
    [
        [
            [1.0,1.0,1.0,1.0],
            [2.0,2.0,3.0,3.0]
        ],
        [
            [4.0,4.0,4.0,4.0],
            [5.0,5.0,6.0,6.0]
        ]
    ], dtype=tf.float32
    )
    
    expected_boxes_cxcywh_batched = tf.constant(
    [
        [
            [1.0,1.0,0.0,0.0],
            [2.5,2.5,1.0,1.0]
        ],
        [
            [4.0,4.0,0.0,0.0],
            [5.5,5.5,1.0,1.0]
        ]
    ], dtype=tf.float32
    )
    
    boxes_cxcywh_batched = xyxy_to_cxcywh_batched(box_xyxy_batched)
    
    tf.debugging.assert_near(boxes_cxcywh_batched, expected_boxes_cxcywh_batched)
    
def test_batched_zero_area_box_cxcywh_to_xyxy():
    box_cxcywh_batched = tf.constant(
    [
        [
            [1.0,1.0,0.0,0.0],
            [2.5,2.5,1.0,1.0]
        ],
        [
            [4.0,4.0,0.0,0.0],
            [5.5,5.5,1.0,1.0]
        ]
    ], dtype=tf.float32
    )
    
    expected_boxes_xyxy_batched = tf.constant(
    [
        [
            [1.0,1.0,1.0,1.0],
            [2.0,2.0,3.0,3.0]
        ],
        [
            [4.0,4.0,4.0,4.0],
            [5.0,5.0,6.0,6.0]
        ]
    ], dtype=tf.float32
    )
    
    boxes_xyxy_batched = cxcywh_toxyxy_batched(box_cxcywh_batched)
    
    tf.debugging.assert_near(boxes_xyxy_batched, expected_boxes_xyxy_batched)
    

# Checking error handling for batched conversions
def test_batched_xyxy_to_cxcywh_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 3, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = xyxy_to_cxcywh_batched(bad_boxes_batched)
        
        
def test_batched_cxcywh_to_xyxy_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = cxcywh_toxyxy_batched(bad_boxes_batched)
        
    
# Testing to_yxyx_core function
def test_to_yxyx_core_simple():
    box_xyxy = tf.constant([[10.0, 20.0, 30.0, 40.0]], dtype=tf.float32)
    expected_box_yxyx = tf.constant([[20.0, 10.0, 40.0, 30.0]], dtype=tf.float32)
    
    box_yxyx = to_yxyx_core(box_xyxy)
    
    tf.debugging.assert_near(box_yxyx, expected_box_yxyx)
    
def test_to_yxyx_batched():
    box_xyxy_batched = tf.constant(
    [
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0]
        ],
        [
            [15.0, 25.0, 35.0, 45.0],
            [55.0, 65.0, 75.0, 85.0]
        ]
    ], dtype=tf.float32
    )
    
    expected_box_yxyx_batched = tf.constant(
    [
        [
            [20.0, 10.0, 40.0, 30.0],
            [60.0, 50.0, 80.0, 70.0]
        ],
        [
            [25.0, 15.0, 45.0, 35.0],
            [65.0, 55.0, 85.0, 75.0]
        ]
    ], dtype=tf.float32
    )
    
    box_yxyx_batched = to_yxyx_batched(box_xyxy_batched)
    
    tf.debugging.assert_near(box_yxyx_batched, expected_box_yxyx_batched)
    
def test_to_yxyx_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = to_yxyx_core(bad_boxes)
        
def test_to_yxyx_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = to_yxyx_batched(bad_boxes_batched)
  
# Testing from_yxyx_core function
def test_from_yxyx_core_simple():
    box_yxyx = tf.constant([[20.0, 10.0, 40.0, 30.0]], dtype=tf.float32)
    expected_box_xyxy = tf.constant([[10.0, 20.0, 30.0, 40.0]], dtype=tf.float32)
    
    box_xyxy = from_yxyx_core(box_yxyx)
    
    tf.debugging.assert_near(box_xyxy, expected_box_xyxy)
    
def test_from_yxyx_batched():
    box_yxyx_batched = tf.constant(
    [
        [
            [20.0, 10.0, 40.0, 30.0],
            [60.0, 50.0, 80.0, 70.0]
        ],
        [
            [25.0, 15.0, 45.0, 35.0],
            [65.0, 55.0, 85.0, 75.0]
        ]
    ], dtype=tf.float32
    )
    
    expected_box_xyxy_batched = tf.constant(
    [
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0]
        ],
        [
            [15.0, 25.0, 35.0, 45.0],
            [55.0, 65.0, 75.0, 85.0]
        ]
    ], dtype=tf.float32
    )
    
    box_xyxy_batched = from_yxyx_batched(box_yxyx_batched)
    
    tf.debugging.assert_near(box_xyxy_batched, expected_box_xyxy_batched)
    
def test_from_yxyx_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = from_yxyx_core(bad_boxes)
        
def test_from_yxyx_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = from_yxyx_batched(bad_boxes_batched)
        
# Testing round trip for to_yxyx and from_yxyx
def test_round_trip_to_from_yxyx_core():
    original_box_xyxy = tf.constant([[10.0, 20.0, 30.0, 40.0]], dtype=tf.float32)
    
    box_yxyx = to_yxyx_core(original_box_xyxy)
    round_trip_box_xyxy = from_yxyx_core(box_yxyx)
    
    tf.debugging.assert_near(round_trip_box_xyxy, original_box_xyxy)
    
def test_round_trip_to_from_yxyx_batched():
    original_box_xyxy_batched = tf.constant(
    [
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0]
        ],
        [
            [15.0, 25.0, 35.0, 45.0],
            [55.0, 65.0, 75.0, 85.0]
        ]
    ], dtype=tf.float32
    )
    
    box_yxyx_batched = to_yxyx_batched(original_box_xyxy_batched)
    round_trip_box_xyxy_batched = from_yxyx_batched(box_yxyx_batched)
    
    tf.debugging.assert_near(round_trip_box_xyxy_batched, original_box_xyxy_batched)
    
#  Testing clip_xyxy_core function
def test_clip_xyxy_core_simple():
    box_xyxy = tf.constant([[-10.0, 20.0, 130.0, 250.0]], dtype=tf.float32)
    expected_clipped_box = tf.constant([[0.0, 20.0, 130.0, 100.0]], dtype=tf.float32)
    
    clipped_box = clip_xyxy_core(box_xyxy, 100.0, 200.0)
    
    tf.debugging.assert_near(clipped_box, expected_clipped_box)
    
def test_clip_xyxy_batched():
    box_xyxy_batched = tf.constant(
    [
        [
            [-10.0, 20.0, 130.0, 250.0],
            [50.0, -30.0, 220.0, 180.0]
        ],
        [
            [15.0, 25.0, 350.0, 450.0],
            [-50.0, 65.0, 75.0, 85.0]
        ]
    ], dtype=tf.float32
    )
    
    expected_clipped_box_batched = tf.constant(
    [
        [
            [0.0, 20.0, 130.0, 100.0],
            [50.0, 0.0, 200.0, 100.0]
        ],
        [
            [15.0, 25.0, 200.0, 100.0],
            [0.0, 65.0, 75.0, 85.0]
        ]
    ], dtype=tf.float32
    )
    
    clipped_box_batched = clip_xyxy_batched(box_xyxy_batched, 100.0, 200.0)
    
    tf.debugging.assert_near(clipped_box_batched, expected_clipped_box_batched)
    
def test_clip_xyxy_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = clip_xyxy_core(bad_boxes, 100.0, 200.0)
        
def test_clip_xyxy_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = clip_xyxy_batched(bad_boxes_batched, 100.0, 200.0)
        
def test_clip_xyxy_core_negative_dimensions():
    box_xyxy = tf.constant([[10.0, 20.0, 30.0, 40.0]], dtype=tf.float32)

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = clip_xyxy_core(box_xyxy, -100.0, 200.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = clip_xyxy_core(box_xyxy, 100.0, -200.0)
        
def test_clip_xyxy_batched_negative_dimensions():
    box_xyxy_batched = tf.constant(
    [
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0]
        ]
    ], dtype=tf.float32
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = clip_xyxy_batched(box_xyxy_batched, -100.0, 200.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = clip_xyxy_batched(box_xyxy_batched, 100.0, -200.0)
        
# Testing abs_to_rel_xyxy_core function
def test_abs_to_rel_xyxy_core_simple():
    box_xyxy = tf.constant([[50.0, 100.0, 150.0, 200.0]], dtype=tf.float32)
    image_width = 200.0
    image_height = 400.0
    expected_rel_box = tf.constant([[0.25, 0.25, 0.75, 0.5]], dtype=tf.float32)
    
    rel_box = abs_to_rel_xyxy_core(box_xyxy, image_height,image_width )
    
    tf.debugging.assert_near(rel_box, expected_rel_box)
    
def test_abs_to_rel_xyxy_batched():
    box_xyxy_batched = tf.constant(
    [
        [
            [50.0, 100.0, 150.0, 200.0],
            [0.0, 0.0, 200.0, 400.0]
        ],
        [
            [25.0, 50.0, 175.0, 350.0],
            [100.0, 200.0, 150.0, 300.0]
        ]
    ], dtype=tf.float32
    )
    image_width = 200.0
    image_height = 400.0
    
    expected_rel_box_batched = tf.constant(
    [
        [
            [0.25, 0.25, 0.75, 0.5],
            [0.0, 0.0, 1.0, 1.0]
        ],
        [
            [0.125, 0.125, 0.875, 0.875],
            [0.5, 0.5, 0.75, 0.75]
        ]
    ], dtype=tf.float32
    )
    
    rel_box_batched = abs_to_rel_xyxy_batched(box_xyxy_batched, image_height, image_width)
    
    tf.debugging.assert_near(rel_box_batched, expected_rel_box_batched)
    
def test_abs_to_rel_xyxy_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = abs_to_rel_xyxy_core(bad_boxes, 200.0, 400.0)
        
def test_abs_to_rel_xyxy_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = abs_to_rel_xyxy_batched(bad_boxes_batched, 200.0, 400.0)
        
def test_abs_to_rel_xyxy_core_raises_on_zero_dimensions():
    box_xyxy = tf.constant([[50.0, 100.0, 150.0, 200.0]], dtype=tf.float32)

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = abs_to_rel_xyxy_core(box_xyxy, 0.0, 400.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = abs_to_rel_xyxy_core(box_xyxy, 200.0, 0.0)
        
def test_abs_to_rel_xyxy_batched_raises_on_zero_dimensions():
    box_xyxy_batched = tf.constant(
    [
        [
            [50.0, 100.0, 150.0, 200.0],
            [0.0, 0.0, 200.0, 400.0]
        ]
    ], dtype=tf.float32
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = abs_to_rel_xyxy_batched(box_xyxy_batched, 0.0, 400.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = abs_to_rel_xyxy_batched(box_xyxy_batched, 200.0, 0.0)
        
def test_abs_to_rel_xyxy_core_raises_on_negative_dimensions():
    box_xyxy = tf.constant([[50.0, 100.0, 150.0, 200.0]], dtype=tf.float32)

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = abs_to_rel_xyxy_core(box_xyxy, -200.0, 400.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = abs_to_rel_xyxy_core(box_xyxy, 200.0, -400.0)
        
def test_abs_to_rel_xyxy_batched_raises_on_negative_dimensions():
    box_xyxy_batched = tf.constant(
    [
        [
            [50.0, 100.0, 150.0, 200.0],
            [0.0, 0.0, 200.0, 400.0]
        ]
    ], dtype=tf.float32
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = abs_to_rel_xyxy_batched(box_xyxy_batched, -200.0, 400.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = abs_to_rel_xyxy_batched(box_xyxy_batched, 200.0, -400.0)
        
# Testing rel_to_abs_xyxy_core function
def test_rel_to_abs_xyxy_core_simple():
    box_xyxy = tf.constant([[0.25, 0.25, 0.75, 0.5]], dtype=tf.float32)
    image_width = 200.0
    image_height = 400.0
    expected_abs_box = tf.constant([[50.0, 100.0, 150.0, 200.0]], dtype=tf.float32)
    
    abs_box = rel_to_abs_xyxy_core(box_xyxy, image_height,image_width )
    
    tf.debugging.assert_near(abs_box, expected_abs_box)
    
def test_rel_to_abs_xyxy_batched():
    box_xyxy_batched = tf.constant(
    [
        [
            [0.25, 0.25, 0.75, 0.5],
            [0.0, 0.0, 1.0, 1.0]
        ],
        [
            [0.125, 0.125, 0.875, 0.875],
            [0.5, 0.5, 0.75, 0.75]
        ]
    ], dtype=tf.float32
    )
    image_width = 200.0
    image_height = 400.0
    
    expected_abs_box_batched = tf.constant(
    [
        [
            [50.0, 100.0, 150.0, 200.0],
            [0.0, 0.0, 200.0, 400.0]
        ],
        [
            [25.0, 50.0, 175.0, 350.0],
            [100.0, 200.0, 150.0, 300.0]
        ]
    ], dtype=tf.float32
    )
    
    abs_box_batched = rel_to_abs_xyxy_batched(box_xyxy_batched, image_height, image_width)
    
    tf.debugging.assert_near(abs_box_batched, expected_abs_box_batched)
    
def test_rel_to_abs_xyxy_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = rel_to_abs_xyxy_core(bad_boxes, 200.0, 400.0)
        
def test_rel_to_abs_xyxy_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = rel_to_abs_xyxy_batched(bad_boxes_batched, 200.0, 400.0)
        
def test_rel_to_abs_xyxy_core_raises_on_zero_dimensions():
    box_xyxy = tf.constant([[0.25, 0.25, 0.75, 0.5]], dtype=tf.float32)

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = rel_to_abs_xyxy_core(box_xyxy, 0.0, 400.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = rel_to_abs_xyxy_core(box_xyxy, 200.0, 0.0)
        
def test_rel_to_abs_xyxy_batched_raises_on_zero_dimensions():
    box_xyxy_batched = tf.constant(
    [
        [
            [0.25, 0.25, 0.75, 0.5],
            [0.0, 0.0, 1.0, 1.0]
        ]
    ], dtype=tf.float32
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = rel_to_abs_xyxy_batched(box_xyxy_batched, 0.0, 400.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = rel_to_abs_xyxy_batched(box_xyxy_batched, 200.0, 0.0)
        
def test_rel_to_abs_xyxy_core_raises_on_negative_dimensions():
    box_xyxy = tf.constant([[0.25, 0.25, 0.75, 0.5]], dtype=tf.float32)

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = rel_to_abs_xyxy_core(box_xyxy, -200.0, 400.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = rel_to_abs_xyxy_core(box_xyxy, 200.0, -400.0)
        
def test_rel_to_abs_xyxy_batched_raises_on_negative_dimensions():
    box_xyxy_batched = tf.constant(
    [
        [
            [0.25, 0.25, 0.75, 0.5],
            [0.0, 0.0, 1.0, 1.0]
        ]
    ], dtype=tf.float32
    )

    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = rel_to_abs_xyxy_batched(box_xyxy_batched, -200.0, 400.0)
        
    with pytest.raises(tf.errors.InvalidArgumentError):
        _ = rel_to_abs_xyxy_batched(box_xyxy_batched, 200.0, -400.0)
        
# Testing the Area functions
def test_area_xyxy_core_simple():
    boxes_xyxy = tf.constant(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 20.0],
            [2.0, 3.0, 8.0, 9.0]
        ], dtype=tf.float32
    )
    
    expected_areas = tf.constant([100.0, 150.0, 36.0], dtype=tf.float32)
    
    areas = area_xyxy_core(boxes_xyxy)
    
    tf.debugging.assert_near(areas, expected_areas)
    
def test_area_xyxy_batched():
    boxes_xyxy_batched = tf.constant(
    [
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 20.0]
        ],
        [
            [2.0, 3.0, 8.0, 9.0],
            [1.0, 1.0, 4.0, 5.0]
        ]
    ], dtype=tf.float32
    )
    
    expected_areas_batched = tf.constant(
    [
        [100.0, 150.0],
        [36.0, 12.0]
    ], dtype=tf.float32
    )
    
    areas_batched = area_xyxy_batched(boxes_xyxy_batched)
    
    tf.debugging.assert_near(areas_batched, expected_areas_batched)
    
def test_area_xyxy_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = area_xyxy_core(bad_boxes)
        
def test_area_xyxy_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = area_xyxy_batched(bad_boxes_batched)
        
# Testing the Intersection functions
def test_intersection_xyxy_core_simple():
    boxes1 = tf.constant(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 20.0]
        ], dtype=tf.float32
    )
    
    boxes2 = tf.constant(
        [
            [5.0, 5.0, 15.0, 15.0],
            [0.0, 0.0, 10.0, 10.0]
        ], dtype=tf.float32
    )
    
    expected_intersections = tf.constant(
        [
            [25.0, 100.0],
            [100.0, 25.0],
        ],
        dtype=tf.float32,
    )
    
    intersections = intersection_xyxy_core(boxes1, boxes2)
    
    tf.debugging.assert_near(intersections, expected_intersections)
    
def test_intersection_xyxy_batched():
    
    boxes1_batched = tf.constant(
    [
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 20.0]
        ],
        [
            [2.0, 3.0, 8.0, 9.0],
            [1.0, 1.0, 4.0, 5.0]
        ]
    ], dtype=tf.float32
    )
    
    boxes2_batched = tf.constant(
    [
        [
            [5.0, 5.0, 15.0, 15.0],
            [0.0, 0.0, 10.0, 10.0]
        ],
        [
            [4.0, 4.0, 10.0, 10.0],
            [2.0, 2.0, 5.0, 6.0]
        ]
    ], dtype=tf.float32
    )
    
    expected_intersections_batched = tf.constant(
        [
            [
                [25.0, 100.0],
                [100.0, 25.0],
            ],
            [
                [20.0, 9.0],
                [0.0, 6.0],
            ],
        ],
        dtype=tf.float32,
    )
    
    intersections_batched = intersection_xyxy_batched(boxes1_batched, boxes2_batched)
    
    tf.debugging.assert_near(intersections_batched, expected_intersections_batched)
    
def test_intersection_xyxy_core_raises_on_wrong_last_dim():
    bad_boxes1 = tf.zeros((2, 3), dtype=tf.float32)
    bad_boxes2 = tf.zeros((2, 4), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = intersection_xyxy_core(bad_boxes1, bad_boxes2)
        
def test_intersection_xyxy_batched_raises_on_wrong_last_dim():
    bad_boxes1_batched = tf.zeros((2, 4, 5), dtype=tf.float32)
    bad_boxes2_batched = tf.zeros((2, 3, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = intersection_xyxy_batched(bad_boxes1_batched, bad_boxes2_batched)
        
# Testing the union functions
def test_union_xyxy_core_simple():
    area_1 = tf.constant([100.0, 150.0], dtype=tf.float32)
    area_2 = tf.constant([200.0, 100.0], dtype=tf.float32)
    intersection = tf.constant(
        [
            [25.0, 50.0],
            [100.0, 25.0],
        ],
        dtype=tf.float32,
    )
    
    expected_unions = tf.constant(
        [
            [275.0, 150.0],
            [250.0, 225.0],
        ],
        dtype=tf.float32,
    )
    
    unions = union_from_areas_core(area_1, area_2, intersection)
    
    tf.debugging.assert_near(unions, expected_unions)
    
def test_union_xyxy_batched():
    area_1_batched = tf.constant(
    [
        [100.0, 150.0],
        [36.0, 12.0]
    ], dtype=tf.float32
    )
    
    area_2_batched = tf.constant(
    [
        [200.0, 100.0],
        [60.0, 30.0]
    ], dtype=tf.float32
    )
    
    intersection_batched = tf.constant(
        [
            [
                [25.0, 50.0],
                [100.0, 25.0],
            ],
            [
                [20.0, 9.0],
                [0.0, 6.0],
            ],
        ],
        dtype=tf.float32,
    )
    
    expected_unions_batched = tf.constant(
        [
            [
                [275.0, 150.0],
                [250.0, 225.0],
            ],
            [
                [76.0, 57.0],
                [72.0, 36.0],
            ],
        ],
        dtype=tf.float32,
    )
    
    unions_batched = union_from_areas_batched(area_1_batched, area_2_batched, intersection_batched)
    
    tf.debugging.assert_near(unions_batched, expected_unions_batched)
    
# Testing the IoU functions
def test_iou_xyxy_core_simple():
    boxes1 = tf.constant(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 20.0],
        ],
        dtype=tf.float32,
    )

    boxes2 = tf.constant(
        [
            [5.0, 5.0, 15.0, 15.0],
            [0.0, 0.0, 10.0, 10.0],
        ],
        dtype=tf.float32,
    )

    expected_ious = tf.constant(
        [
            [25.0 / 175.0, 100.0 / 100.0],
            [100.0 / 150.0, 25.0 / 225.0],
        ],
        dtype=tf.float32,
    )

    ious = iou_matrix_core(boxes1, boxes2)

    tf.debugging.assert_near(ious, expected_ious, rtol=1e-6, atol=1e-6)
  
def test_iou_xyxy_batched():
    boxes1_batched = tf.constant(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 5.0, 15.0, 20.0],
            ],
            [
                [2.0, 3.0, 8.0, 9.0],
                [1.0, 1.0, 4.0, 5.0],
            ],
        ],
        dtype=tf.float32,
    )

    boxes2_batched = tf.constant(
        [
            [
                [5.0, 5.0, 15.0, 15.0],
                [0.0, 0.0, 10.0, 10.0],
            ],
            [
                [4.0, 4.0, 10.0, 10.0],
                [2.0, 2.0, 5.0, 6.0],
            ],
        ],
        dtype=tf.float32,
    )

    expected_ious_batched = tf.constant(
        [
            [
                [25.0 / 175.0, 100.0 / 100.0],
                [100.0 / 150.0, 25.0 / 225.0],
            ],
            [
                [20.0 / 52.0, 9.0 / 39.0],
                [0.0 / 48.0, 6.0 / 18.0],
            ],
        ],
        dtype=tf.float32,
    )

    ious_batched = iou_matrix_batched(boxes1_batched, boxes2_batched)

    tf.debugging.assert_near(ious_batched, expected_ious_batched, rtol=1e-6, atol=1e-6)

# Testing the hflip functions
def test_hflip_xyxy_core_simple():
    box_xyxy = tf.constant([[10.0, 20.0, 30.0, 40.0]], dtype=tf.float32)
    image_width = 100.0
    expected_flipped_box = tf.constant([[70.0, 20.0, 90.0, 40.0]], dtype=tf.float32)
    
    flipped_box = hflip_xyxy_core(box_xyxy, image_width)
    
    tf.debugging.assert_near(flipped_box, expected_flipped_box)
    
def test_hflip_xyxy_batched():
    box_xyxy_batched = tf.constant(
    [
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0]
        ],
        [
            [15.0, 25.0, 35.0, 45.0],
            [55.0, 65.0, 75.0, 85.0]
        ]
    ], dtype=tf.float32
    )
    image_width = 100.0
    
    expected_flipped_box_batched = tf.constant(
    [
        [
            [70.0, 20.0, 90.0, 40.0],
            [30.0, 60.0, 50.0, 80.0]
        ],
        [
            [65.0, 25.0, 85.0, 45.0],
            [25.0, 65.0, 45.0, 85.0]
        ]
    ], dtype=tf.float32
    )
    
    flipped_box_batched = hflip_xyxy_batched(box_xyxy_batched, image_width)
    
    tf.debugging.assert_near(flipped_box_batched, expected_flipped_box_batched)
    
def test_hflip_xyxy_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = hflip_xyxy_core(bad_boxes, 100.0)
        
def test_hflip_xyxy_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = hflip_xyxy_batched(bad_boxes_batched, 100.0)
        
        
# Testing the vflip functions
def test_vflip_xyxy_core_simple():
    box_xyxy = tf.constant([[10.0, 20.0, 30.0, 40.0]], dtype=tf.float32)
    image_height = 100.0
    expected_flipped_box = tf.constant([[10.0, 60.0, 30.0, 80.0]], dtype=tf.float32)
    
    flipped_box = vflip_xyxy_core(box_xyxy, image_height)
    
    tf.debugging.assert_near(flipped_box, expected_flipped_box)
    
def test_vflip_xyxy_batched():
    box_xyxy_batched = tf.constant(
    [
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0]
        ],
        [
            [15.0, 25.0, 35.0, 45.0],
            [55.0, 65.0, 75.0, 85.0]
        ]
    ], dtype=tf.float32
    )
    image_height = 100.0
    
    expected_flipped_box_batched = tf.constant(
    [
        [
            [10.0, 60.0, 30.0, 80.0],
            [50.0, 20.0, 70.0, 40.0]
        ],
        [
            [15.0, 55.0, 35.0, 75.0],
            [55.0, 15.0, 75.0, 35.0]
        ]
    ], dtype=tf.float32
    )
    
    flipped_box_batched = vflip_xyxy_batched(box_xyxy_batched, image_height)
    
    tf.debugging.assert_near(flipped_box_batched, expected_flipped_box_batched)
    
def test_vflip_xyxy_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = vflip_xyxy_core(bad_boxes, 100.0)
        
def test_vflip_xyxy_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = vflip_xyxy_batched(bad_boxes_batched, 100.0)
        
# Testing the resize functions
def test_resize_xyxy_core_simple():
    box_xyxy = tf.constant([[10.0, 20.0, 30.0, 40.0]], dtype=tf.float32)
    original_width = 100.0
    original_height = 200.0
    new_width = 200.0
    new_height = 400.0
    expected_resized_box = tf.constant([[20.0, 40.0, 60.0, 80.0]], dtype=tf.float32)
    
    resized_box = resize_xyxy_core(box_xyxy, original_height, original_width, new_height, new_width)
    
    tf.debugging.assert_near(resized_box, expected_resized_box)
    
def test_resize_xyxy_batched():
    box_xyxy_batched = tf.constant(
    [
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0]
        ],
        [
            [15.0, 25.0, 35.0, 45.0],
            [55.0, 65.0, 75.0, 85.0]
        ]
    ], dtype=tf.float32
    )
    original_width = 100.0
    original_height = 200.0
    new_width = 200.0
    new_height = 400.0
    
    expected_resized_box_batched = tf.constant(
    [
        [
            [20.0, 40.0, 60.0, 80.0],
            [100.0, 120.0, 140.0, 160.0]
        ],
        [
            [30.0, 50.0, 70.0, 90.0],
            [110.0, 130.0, 150.0, 170.0]
        ]
    ], dtype=tf.float32
    )
    
    resized_box_batched = resize_xyxy_batched(box_xyxy_batched, original_height, original_width, new_height, new_width)
    
    tf.debugging.assert_near(resized_box_batched, expected_resized_box_batched)
    
def test_resize_xyxy_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = resize_xyxy_core(bad_boxes, 200.0, 100.0, 400.0, 200.0)
        
def test_resize_xyxy_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = resize_xyxy_batched(bad_boxes_batched, 200.0, 100.0, 400.0, 200.0)
    
# Testing the crop functions
def test_crop_xyxy_core_simple():
    box_xyxy = tf.constant([[10.0, 20.0, 30.0, 40.0]], dtype=tf.float32)
    crop_xmin = 5.0
    crop_ymin = 10.0
    crop_xmax = 100.0   # wide enough so no clipping on the right
    crop_ymax = 100.0   # tall enough so no clipping at bottom

    expected_cropped_box = tf.constant([[5.0, 10.0, 25.0, 30.0]], dtype=tf.float32)

    cropped_box = crop_xyxy_core(
        box_xyxy,
        crop_xmin,  # x first
        crop_ymin,
        crop_xmax,
        crop_ymax,
    )

    tf.debugging.assert_near(cropped_box, expected_cropped_box)
    
def test_crop_xyxy_batched():
    box_xyxy_batched = tf.constant(
        [
            [
                [10.0, 20.0, 30.0, 40.0],
                [50.0, 60.0, 70.0, 80.0],
            ],
            [
                [15.0, 25.0, 35.0, 45.0],
                [55.0, 65.0, 75.0, 85.0],
            ],
        ],
        dtype=tf.float32,
    )
    crop_xmin = 5.0
    crop_ymin = 10.0
    crop_xmax = 100.0
    crop_ymax = 100.0

    expected_cropped_box_batched = tf.constant(
        [
            [
                [5.0, 10.0, 25.0, 30.0],
                [45.0, 50.0, 65.0, 70.0],
            ],
            [
                [10.0, 15.0, 30.0, 35.0],
                [50.0, 55.0, 70.0, 75.0],
            ],
        ],
        dtype=tf.float32,
    )

    cropped_box_batched = crop_xyxy_batched(
        box_xyxy_batched, crop_xmin, crop_ymin, crop_xmax, crop_ymax
    )

    tf.debugging.assert_near(cropped_box_batched, expected_cropped_box_batched)


    
def test_crop_xyxy_core_raises_on_wrong_last_dim():
    bad_boxes = tf.zeros((2, 3), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = crop_xyxy_core(bad_boxes, 10.0, 5.0,100, 200.0)
        
def test_crop_xyxy_batched_raises_on_wrong_last_dim():
    bad_boxes_batched = tf.zeros((2, 4, 5), dtype=tf.float32)

    with pytest.raises(TypeError, match="Can not cast TensorSpec"):
        _ = crop_xyxy_batched(bad_boxes_batched, 10.0, 5.0, 100, 200.0)   
    