import tensorflow as tf

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32),
    ]
)
def xyxy_to_cxcywh_core(boxes : tf.Tensor):

    # Checking the type
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    
    # Converting XY- Coordinate boxes into Center format
    x_min, y_min, x_max, y_max = tf.split(boxes,num_or_size_splits = 4, axis=-1)

    cx = (x_max + x_min)/2
    cy = (y_max + y_min)/2

    w = x_max - x_min
    h = y_max - y_min

    return tf.concat([cx,cy,w,h], axis = -1)

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None,None, 4], dtype = tf.float32),
    ]
)
def xyxy_to_cxcywh_batched(boxes: tf.Tensor):
    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]
    
    flattened_boxes = tf.reshape(boxes,[-1,4])
    flattened_converted_boxes = xyxy_to_cxcywh_core(flattened_boxes)

    return tf.reshape(flattened_converted_boxes, [B,N,4])

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32),
    ]
)
def cxcywh_toxyxy_core(boxes: tf.Tensor):

    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    
    cx, cy, w, h = tf.split(boxes,num_or_size_splits = 4, axis=-1)

    x_min = cx - (0.5*w)
    y_min = cy - (0.5*h)

    x_max = cx + (0.5*w)
    y_max = cy + (0.5*h)

    return tf.concat([x_min,y_min,x_max,y_max], axis = -1)

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None,None, 4], dtype = tf.float32),
    ]
)
def cxcywh_toxyxy_batched(boxes: tf.Tensor):
    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]
    
    flattened_boxes = tf.reshape(boxes,[-1,4])
    flattened_converted_boxes = cxcywh_toxyxy_core(flattened_boxes)

    return tf.reshape(flattened_converted_boxes, [B,N,4])

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32),
    ]
)
def to_yxyx_core(boxes_xyxy: tf.Tensor):
    
    tf.debugging.assert_equal(tf.shape(boxes_xyxy)[-1], 4, message="boxes last dim must be 4")
    
    x_min,y_min,x_max,y_max = tf.split(boxes_xyxy,num_or_size_splits = 4, axis=-1)

    return tf.concat([y_min,x_min,y_max,x_max],axis=-1)

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None,None, 4], dtype = tf.float32),
    ]
)
def to_yxyx_batched(boxes_xyxy: tf.Tensor):
    B = tf.shape(boxes_xyxy)[0]
    N = tf.shape(boxes_xyxy)[1]
    
    flattened_boxes = tf.reshape(boxes_xyxy,[-1,4])
    flattened_converted_boxes = to_yxyx_core(flattened_boxes)

    return tf.reshape(flattened_converted_boxes, [B,N,4])

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32),
    ]
)
def from_yxyx_core(boxes_yxyx: tf.Tensor):
    
    tf.debugging.assert_equal(tf.shape(boxes_yxyx)[-1], 4, message="boxes last dim must be 4")
    
    y_min,x_min,y_max,x_max = tf.split(boxes_yxyx,num_or_size_splits = 4, axis=-1)

    return tf.concat([x_min,y_min,x_max,y_max],axis=-1)

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None,None, 4], dtype = tf.float32),
    ]
)
def from_yxyx_batched(boxes_yxyx: tf.Tensor):
    B = tf.shape(boxes_yxyx)[0]
    N = tf.shape(boxes_yxyx)[1]
    
    flattened_boxes = tf.reshape(boxes_yxyx,[-1,4])
    flattened_converted_boxes = from_yxyx_core(flattened_boxes)

    return tf.reshape(flattened_converted_boxes, [B,N,4])

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32)
    ]
)
def clip_xyxy_core(boxes: tf.Tensor, H: tf.float32, W: tf.float32):

    tf.debugging.assert_greater(H, 0.0, message="Height must be more than 0")
    tf.debugging.assert_greater(W, 0.0, message="Height must be more than 0")
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    
    x_min,y_min,x_max,y_max = tf.split(boxes,num_or_size_splits = 4, axis=-1)

    # Clip the boxes between [0,W] & [0,H]
    x_min = tf.clip_by_value(x_min, 0, W)
    y_min = tf.clip_by_value(y_min, 0, H)

    x_max = tf.clip_by_value(x_max, 0, W)
    y_max = tf.clip_by_value(y_max, 0, H)

    return tf.concat([x_min,y_min,x_max,y_max],axis=-1)

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None,None, 4], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32)
    ]
)
def clip_xyxy_batched(boxes: tf.Tensor, H: tf.float32, W: tf.float32):
    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]
    
    flattened_boxes = tf.reshape(boxes,[-1,4])
    flattened_converted_boxes = clip_xyxy_core(flattened_boxes,H,W)

    return tf.reshape(flattened_converted_boxes, [B,N,4])

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32)
    ]
)
def abs_to_rel_xyxy_core(boxes: tf.Tensor, H: tf.float32 ,W: tf.float32):
    # Need to check the shape
    tf.debugging.assert_greater(H, 0.0, message="Height must be more than 0")
    tf.debugging.assert_greater(W, 0.0, message="Height must be more than 0")
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")

    clipped_boxes_xyxy = clip_xyxy_core(boxes,H,W)

    x_min,y_min,x_max,y_max = tf.split(clipped_boxes_xyxy,num_or_size_splits = 4, axis=-1)

    # Scaling the values
    x_min = x_min / W
    y_min = y_min / H

    x_max = x_max / W
    y_max = y_max / H

    return tf.concat([x_min,y_min,x_max,y_max],axis=-1)
    
@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, None, 4], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32)
    ]
)
def abs_to_rel_xyxy_batched(boxes: tf.Tensor, H: tf.float32, W: tf.float32):
    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]

    flattened_boxes_xyxy =  tf.reshape(boxes,[-1,4])
    normalized_boxes_xyxy = abs_to_rel_xyxy_core(flattened_boxes_xyxy, H, W)

    return tf.reshape(normalized_boxes_xyxy,[B,N,4])

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32)
    ]
)
def rel_to_abs_xyxy_core(boxes: tf.Tensor, H: tf.float32 ,W: tf.float32):
    # Need to check the shape
    tf.debugging.assert_greater(H, 0.0, message="Height must be more than 0")
    tf.debugging.assert_greater(W, 0.0, message="Height must be more than 0")
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")

    x_min,y_min,x_max,y_max = tf.split(boxes,num_or_size_splits = 4, axis=-1)

    # Scaling the values
    x_min = x_min * W
    y_min = y_min * H

    x_max = x_max * W
    y_max = y_max * H

    return tf.concat([x_min,y_min,x_max,y_max],axis=-1)
    
@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, None, 4], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32),
        tf.TensorSpec(shape=[], dtype = tf.float32)
    ]
)
def rel_to_abs_xyxy_batched(boxes: tf.Tensor, H: tf.float32 ,W: tf.float32):
    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]

    flattened_boxes = tf.reshape(boxes,[-1,4])
    denormalized_boxes = rel_to_abs_xyxy_core(flattened_boxes,H,W)

    return tf.reshape(denormalized_boxes,[B,N,4])

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32),
    ]
)
def area_xyxy_core(boxes: tf.Tensor):
    # Making sure the shape is correct
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    
    # Split the coordinates
    x_min, y_min, x_max, y_max = tf.split(boxes,num_or_size_splits = 4, axis=-1)

    # Calculate the area
    w = tf.maximum(x_max - x_min, 0.0)
    h = tf.maximum(y_max - y_min, 0.0)

    area = w * h

    return tf.squeeze(area, axis=-1)

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, None, 4], dtype = tf.float32),
    ]
)
def area_xyxy_batched(boxes: tf.Tensor):

    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]

    # Flattening the boxes
    flattened_boxes_xyxy = tf.reshape(boxes,[-1,4])
    flattened_area = area_xyxy_core(flattened_boxes_xyxy)

    return tf.reshape(flattened_area, [B,N])

@tf.function(
    input_signature = [
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32),
        tf.TensorSpec(shape=[None, 4], dtype = tf.float32)
    ]
)
def intersection_xyxy_core(boxes_1,boxes_2):
    tf.debugging.assert_equal(tf.shape(boxes_1)[-1], 4, message="boxes 1 last dim must be 4")
    tf.debugging.assert_equal(tf.shape(boxes_2)[-1], 4, message="boxes 2 last dim must be 4")

    # Split the coordinates
    ax_min,ay_min,ax_max,ay_max = tf.split(boxes_1,num_or_size_splits = 4, axis=-1)
    bx_min,by_min,bx_max,by_max = tf.split(boxes_2,num_or_size_splits = 4, axis=-1)

    # Calculating the proper coordinates
    x_min = tf.maximum(ax_min[:,None], bx_min[None,:])
    y_min = tf.maximum(ay_min[:,None], by_min[None,:])
    x_max = tf.minimum(ax_max[:,None], bx_max[None,:])
    y_max = tf.minimum(ay_max[:,None], by_max[None,:])

    # Calculating the intersection
    w = tf.maximum(x_max - x_min, 0.0)
    h = tf.maximum(y_max - y_min, 0.0)

    intersection = w * h

    return tf.squeeze(intersection,axis=-1)

@tf.function(
    input_signature=[
        tf.TensorSpec([None, None, 4], tf.float32), 
        tf.TensorSpec([None, None, 4], tf.float32),  
    ]
)
def intersection_xyxy_batched(boxes_1, boxes_2):
    
    tf.debugging.assert_equal(tf.shape(boxes_1)[-1], 4, message="boxes_1 last dim must be 4")
    tf.debugging.assert_equal(tf.shape(boxes_2)[-1], 4, message="boxes_2 last dim must be 4")

    a_xmin, a_ymin, a_xmax, a_ymax = tf.split(boxes_1, 4, axis=-1)  
    b_xmin, b_ymin, b_xmax, b_ymax = tf.split(boxes_2, 4, axis=-1) 

    x1 = tf.maximum(a_xmin[:, :, None, :], b_xmin[:, None, :, :])  
    y1 = tf.maximum(a_ymin[:, :, None, :], b_ymin[:, None, :, :])  
    x2 = tf.minimum(a_xmax[:, :, None, :], b_xmax[:, None, :, :])  
    y2 = tf.minimum(a_ymax[:, :, None, :], b_ymax[:, None, :, :])  

    w = tf.maximum(x2 - x1, 0.0)
    h = tf.maximum(y2 - y1, 0.0)
    inter = w * h                                                

    return tf.squeeze(inter, axis=-1)

@tf.function(
    input_signature=[
        tf.TensorSpec([None], tf.float32),     
        tf.TensorSpec([None], tf.float32),     
        tf.TensorSpec([None, None], tf.float32)
    ]
)
def union_from_areas_core(a_area, b_area, inter):
    # Broadcast: (N,1) + (1,M) - (N,M)
    union = a_area[:, None] + b_area[None, :] - inter
    return tf.maximum(union, tf.constant(1e-7, tf.float32)) 

@tf.function(
    input_signature=[
        tf.TensorSpec([None, None], tf.float32),   
        tf.TensorSpec([None, None], tf.float32),   
        tf.TensorSpec([None, None, None], tf.float32)  
    ]
)
def union_from_areas_batched(a_area, b_area, inter):
    union = a_area[:, :, None] + b_area[:, None, :] - inter
    return tf.maximum(union, tf.constant(1e-7, tf.float32))

@tf.function(
    input_signature=[
        tf.TensorSpec([None, 4], tf.float32), 
        tf.TensorSpec([None, 4], tf.float32),  
    ]
)
def iou_matrix_core(boxes_1, boxes_2):
    tf.debugging.assert_rank(boxes_1, 2, message="boxes1 must be (M,4)")
    tf.debugging.assert_equal(tf.shape(boxes_1)[-1], 4)
    tf.debugging.assert_rank(boxes_2, 2, message="boxes2 must be (N,4)")
    tf.debugging.assert_equal(tf.shape(boxes_2)[-1], 4)
    # areas
    
    a_area = area_xyxy_core(boxes_1)
    b_area = area_xyxy_core(boxes_2)
    # intersections
    inter = intersection_xyxy_core(boxes_1, boxes_2)
    # unions
    union = union_from_areas_core(a_area, b_area, inter)
    # IoU
    return inter / union

@tf.function(
    input_signature=[
        tf.TensorSpec([None, None, 4], tf.float32), 
        tf.TensorSpec([None, None, 4], tf.float32), 
    ]
)
def iou_matrix_batched(boxes_1, boxes_2):
    # areas
    a_area = area_xyxy_batched(boxes_1)
    b_area = area_xyxy_batched(boxes_2)
    # intersections
    inter = intersection_xyxy_batched(boxes_1, boxes_2)
    # unions
    union = union_from_areas_batched(a_area, b_area, inter)
    # IoU
    return inter / union

@tf.function(
    input_signature=[
        tf.TensorSpec([None, 4], tf.float32), 
        tf.TensorSpec([], tf.float32),  
    ]
)
def hflip_xyxy_core(boxes: tf.Tensor,W: tf.float32):
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    
    x_min, y_min, x_max, y_max = tf.split(boxes,num_or_size_splits = 4, axis=-1)

    # Flipping the Coordinates
    x_min_new = W - x_max
    x_max_new = W - x_min

    return tf.concat([x_min_new,y_min,x_max_new,y_max],axis=-1)

@tf.function(
    input_signature=[
        tf.TensorSpec([None,None, 4], tf.float32), 
        tf.TensorSpec([], tf.float32),  
    ]
)
def hflip_xyxy_batched(boxes: tf.Tensor,W: tf.float32):
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    tf.debugging.assert_greater(W, 0.0, message="boxes last dim must be 4")
    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]

    flattened_boxes = tf.reshape(boxes,[-1,4])
    flipped_boxes = hflip_xyxy_core(flattened_boxes,W)

    return tf.reshape(flipped_boxes,[B,N,4])

@tf.function(
    input_signature=[
        tf.TensorSpec([None, 4], tf.float32), 
        tf.TensorSpec([], tf.float32),  
    ]
)
def vflip_xyxy_core(boxes: tf.Tensor,H: tf.float32):
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    
    x_min, y_min, x_max, y_max = tf.split(boxes,num_or_size_splits = 4, axis=-1)

    # Flipping the Coordinates
    y_min_new = H - y_max
    y_max_new = H - y_min

    return tf.concat([x_min,y_min_new,x_max,y_max_new],axis=-1)

@tf.function(
    input_signature=[
        tf.TensorSpec([None,None, 4], tf.float32), 
        tf.TensorSpec([], tf.float32),  
    ]
)
def vflip_xyxy_batched(boxes: tf.Tensor,H: tf.float32):
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    tf.debugging.assert_greater(H, 0.0, message="boxes last dim must be 4")
    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]

    flattened_boxes = tf.reshape(boxes,[-1,4])
    flipped_boxes = vflip_xyxy_core(flattened_boxes,H)

    return tf.reshape(flipped_boxes,[B,N,4])

@tf.function(
    input_signature=[
        tf.TensorSpec([None, 4], tf.float32), 
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32)
    ]
)
def resize_xyxy_core(boxes,source_height: tf.float32,source_width: tf.float32,destination_height: tf.float32,destination_width: tf.float32):
    
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    
    s_x = source_width / destination_width
    s_y = destination_height / source_height

    x_min, y_min, x_max, y_max = tf.split(boxes,num_or_size_splits = 4, axis=-1)

    x_min = x_min * s_x
    y_min = y_min * s_y
    x_max = x_max * s_x
    y_max = y_max * s_y

    return tf.concat([x_min,y_min,x_max,y_max],axis=-1)

def resize_xyxy_batched(boxes,source_height: tf.float32,source_width: tf.float32,destination_height: tf.float32,destination_width: tf.float32):
    
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")
    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]

    flattened_boxes = tf.reshape(boxes,[-1,4])
    flipped_boxes = resize_xyxy_core(flattened_boxes,source_height,source_width,destination_height,destination_width)

    return tf.reshape(flipped_boxes,[B,N,4])

@tf.function(
    input_signature=[
        tf.TensorSpec([None, 4], tf.float32), 
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32)
    ]
)
def crop_xyxy_core(boxes,crop_xmin,crop_ymin,crop_xmax,crop_ymax):
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")

    crop_w = crop_xmax - crop_xmin
    crop_h = crop_ymax - crop_ymin

    x_min, y_min, x_max, y_max = tf.split(boxes,num_or_size_splits = 4, axis=-1)

    x_min = tf.minimum(crop_w, tf.maximum(0.0, x_min - crop_xmin))
    y_min = tf.minimum(crop_h, tf.maximum(0.0, y_min - crop_ymin))

    x_max = tf.minimum(crop_w, tf.maximum(0.0, x_max - crop_xmin))
    y_max = tf.minimum(crop_h, tf.maximum(0.0, y_max - crop_ymin))

    return tf.concat([x_min,y_min,x_max,y_max],axis=-1)

@tf.function(
    input_signature=[
        tf.TensorSpec([None,None, 4], tf.float32), 
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.float32)
    ]
)
def crop_xyxy_batched(boxes,crop_xmin,crop_ymin,crop_xmax,crop_ymax):
    tf.debugging.assert_equal(tf.shape(boxes)[-1], 4, message="boxes last dim must be 4")

    B = tf.shape(boxes)[0]
    N = tf.shape(boxes)[1]

    flattened_boxes = tf.reshape(boxes,[-1,4])
    cropped_boxes = crop_xyxy_core(flattened_boxes,crop_xmin,crop_ymin,crop_xmax,crop_ymax)

    return tf.reshape(cropped_boxes,[B,N,4])