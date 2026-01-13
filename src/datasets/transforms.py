import tensorflow as tf
from typing import Any

# Standardization Functions
class ToFloat32:
    def __init__(self, image_precision: tf.dtypes.DType = tf.float32, target_precision:tf.dtypes.DType = tf.float16):
        if not isinstance(image_precision, tf.dtypes.DType) or not isinstance(target_precision, tf.dtypes.DType):
            raise ValueError("The precision is not a Tensorflow dtype")
        
        # Float Precision Options    
        self._image_dtype = image_precision
        self._target_dtype = target_precision

    def __call__(self, image, target):
       # Converting the image from uint8 to float32
        image = tf.cast(image, dtype = self._image_dtype)
        target['boxes'] = tf.constant(target['boxes'], dtype = self._target_dtype)
        target['labels'] = tf.cast(target['labels'], dtype = tf.int32)

        return image, target
    
class Scale01:
    def __init__(self):
        pass
    def __call__(self, image, target):
        # Scale the image
        image = image / 255
            
        return image, target
    
class NormalizeBoundingBoxes:
    def __init__(self):
        pass

    def __call__(self, image, target):
        # Scale the Boxes
        width, height = target['orig_size']
        x1, y1, x2, y2 = tf.split(target['boxes'], num_or_size_splits = 4, axis = -1)

        x1 = x1 / tf.cast(height,dtype = target['boxes'].dtype)
        y1 = y1 / tf.cast(width,dtype = target['boxes'].dtype)
        x2 = x2 / tf.cast(height,dtype = target['boxes'].dtype)
        y2 = y2 / tf.cast(width,dtype = target['boxes'].dtype)

        boxes = tf.concat([ x1, y1, x2, y2],axis=-1)
        target['boxes'] = boxes
            
        return image, target

class Compose:
    def __init__(self, transforms: list[Any]):
        self._transforms = transforms

    def __call__(self, image, target):
        for transform in self._transforms:
            image, target = transform(image,target)

            if image is None:
                raise RuntimeError(f"{transform} returned image=None")

            if target is None or "boxes" not in target or "labels" not in target:
                raise RuntimeError(f"{transform} returned invalid target")

        return image, target

class PhotometricDistort:
    def __init__(self, p: float =1.0, brightness_delta: float = 0.125, contrast_range: tuple[float, float] = (0.5, 1.5), saturation_range: tuple[float, float] = (0.5,1.5), hue_delta: float = 0.05, channel_swap: bool = False, seed: int| None = None):
        self._p = p
        self._brightness_delta = brightness_delta
        self._contrast_range = tuple(contrast_range)
        self._saturation_range = tuple(saturation_range)
        self._hue_delta = hue_delta
        self._channel_swap = channel_swap
        self._seed = seed
        
    def __call__(self, image, target):
        # Call early exit

        image = self.to_float(image)

        if self._p < 1.0:
            random_num = tf.random.uniform([])
            image = tf.cond(random_num < self._p, lambda: self.distort_image(image), lambda: image)
        else:
            image = self.distort_image(image)

        return image, target

    def distort_image(self, image):
        image = self.determine_outcome(lambda x: tf.image.random_brightness(x, self._brightness_delta,seed = self._seed), image)

        contrast_order = tf.random.uniform([], 0.0, 1.0, seed=self._seed) < 0.5

        image = tf.cond(contrast_order, lambda: self.contrast_first(image), lambda: self.contrast_last(image))

        if self._channel_swap:
            image = self.determine_outcome(self.random_channel_swap, image)

        return image

    def determine_outcome(self, function, image):
        random_num = tf.random.uniform([], 0.0, 1.0, seed=self._seed)
        return tf.cond(random_num < 0.5, lambda: function(image), lambda: image)
    
    def contrast_first(self,image):
        image = self.determine_outcome(lambda x: tf.image.random_contrast(x, self._contrast_range[0], self._contrast_range[1], seed=self._seed), image)
        image = self.determine_outcome(lambda x: tf.image.random_saturation(x, self._saturation_range[0], self._saturation_range[1], seed=self._seed), image)
        image = self.determine_outcome(lambda x: tf.image.random_hue(x, self._hue_delta, seed=self._seed), image)

        return image

    def contrast_last(self,image):
        image = self.determine_outcome(lambda x: tf.image.random_saturation(x, self._saturation_range[0], self._saturation_range[1], seed=self._seed), image)
        image = self.determine_outcome(lambda x: tf.image.random_hue(x, self._hue_delta, seed=self._seed), image)
        image = self.determine_outcome(lambda x: tf.image.random_contrast(x, self._contrast_range[0], self._contrast_range[1], seed=self._seed), image)

        return image

    def random_channel_swap(self, image):
        perm = tf.random.shuffle(tf.constant([0, 1, 2]), seed=self._seed)
        return tf.gather(image, perm, axis=-1)

    def to_float(self, image):
        return tf.cast(image, dtype = tf.float32)

    def determine_outcome(self, function, image):
        random_num = tf.random.uniform([], 0.0, 1.0, seed=self._seed)
        return tf.cond(random_num < 0.5, lambda: function(image), lambda: image)

    def contrast_first(self, image):
        image = self.determine_outcome(lambda x: tf.image.random_contrast(x, self._contrast_range[0], self._contrast_range[1], seed=self.seed), image)
        image = self.determine_outcome(lambda x: tf.image.random_saturation(x, self._saturation_range[0], self._saturation_range[1], seed=self.seed), image)
        image = self.determine_outcome(lambda x: tf.image.random_hue(x, self.hue_delta, seed=self.seed), image)

        return image

    def contrast_last(self, image):
        image = self.determine_outcome(lambda x: tf.image.random_saturation(x, self._saturation_range[0], self._saturation_range[1], seed=self.seed), image)
        image = self.determine_outcome(lambda x: tf.image.random_hue(x, self.hue_delta, seed=self.seed), image)
        image = self.determine_outcome(lambda x: tf.image.random_contrast(x, self._contrast_range[0], self._contrast_range[1], seed=self.seed), image)

        return image

    def random_channel_swap(self, image):
        perm = tf.random.shuffle(tf.constant([0, 1, 2]), seed=self.seed)
        return tf.gather(image, perm, axis=-1)

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self._p = tf.constant(p, dtype = tf.float32)

    def __call__(self, image, target):
        random_num = tf.random.uniform([])

        image, target = tf.cond(random_num < self._p, lambda: self.horizontal_flip(image,target), lambda: self.dont_flip(image, target))

        return image, target
        
    @staticmethod
    def horizontal_flip(image,target):
        shape = tf.shape(image)
        W = tf.cast(shape[1], tf.float32)

        boxes = target['boxes']
        W = tf.cast(W, dtype = boxes.dtype)
        
        x_min, y_min, x_max, y_max = tf.split(boxes,num_or_size_splits = 4, axis=-1)

        # Flipping the Coordinates
        x_min_new = W - x_max
        x_max_new = W - x_min

        boxes = tf.concat([x_min_new,y_min,x_max_new,y_max],axis=-1)

        target['boxes'] = boxes

        # Flipping the image
        image_flipped = tf.image.flip_left_right(image)

        return image_flipped, target
    @staticmethod
    def dont_flip(image, target):
        return image, target
    
class Resize:
    def __init__(self, size: tuple[float, float] | int, mode="stretch"):
        if isinstance(size, int):
            self._size = (size, size)
        elif isinstance(size, tuple) and len(size) == 2:
            self._size = (int(size[0]), int(size[1]))
        else:
            raise ValueError("size must be int or tuple(h, w)")

        if mode not in ("stretch", "letterbox"):
            raise ValueError("mode must be 'stretch' or 'letterbox'")

        self._mode = mode

    def __call__(self, image, target):
        
        shape = tf.shape(image)
        H = tf.cast(shape[0], tf.float32)
        W = tf.cast(shape[1], tf.float32)

        new_h = tf.cast(self._size[0], tf.float32)
        new_w = tf.cast(self._size[1], tf.float32)

        boxes = tf.cast(target["boxes"], tf.float32)
        
        if self._mode == "stretch":
            scaled_x = new_w / W
            scaled_y = new_h / H 

            x1, y1, x2, y2 = tf.split(boxes, num_or_size_splits = 4, axis= -1)

            x1 = x1 * scaled_x
            y1 = y1 * scaled_y
            x2 = x2 * scaled_x
            y2 = y2 * scaled_y

            boxes = tf.concat([x1, y1, x2, y2], axis= -1)

            # Resizing image
            target_size = [self._size[0], self._size[1]]
            image = tf.image.resize(image, target_size)
            
        elif self._mode == "letterbox":
            scale = tf.minimum(new_h / H, new_w / W)
            
            resized_width = tf.cast(tf.floor(W * scale), tf.int32)
            resized_height = tf.cast(tf.floor(H * scale), tf.int32)

            # Resizing image
            image_resized = tf.image.resize(image, [resized_height, resized_width])

            padding_width = tf.cast(self._size[1], tf.int32) - resized_width
            padding_height = tf.cast(self._size[0], tf.int32) - resized_height

            pad_left = padding_width // 2
            pad_right = padding_width - pad_left
            pad_top = padding_height // 2
            pad_bottom = padding_height - pad_top

            image = tf.pad(image_resized, paddings = [[pad_top, pad_bottom],[pad_left, pad_right], [0,0]], mode= "CONSTANT", constant_values = 0.0)

            pad_x = tf.cast(pad_left, tf.float32)
            pad_y = tf.cast(pad_top, tf.float32)

            # Update box coordinates
            x1, y1, x2, y2 = tf.split(boxes, num_or_size_splits = 4, axis= -1)

            x1 = x1 * scale + pad_x
            y1 = y1 * scale + pad_y
            x2 = x2 * scale + pad_x
            y2 = y2 * scale + pad_y

            boxes = tf.concat([x1, y1, x2, y2], axis= -1)
            
        else:
            raise ValueError("")

        x1, y1, x2, y2 = tf.split(boxes, num_or_size_splits = 4, axis= -1)
        
        x1 = tf.clip_by_value(x1, 0.0, new_w)
        y1 = tf.clip_by_value(y1, 0.0, new_h)
        x2 = tf.clip_by_value(x2, 0.0, new_w)
        y2 = tf.clip_by_value(y2, 0.0, new_h)

        boxes = tf.concat([x1, y1, x2, y2], axis= -1)

        target['boxes'] = boxes

        target['resize_info'] = tf.constant(tf.cast([new_h, new_w], tf.float32),dtype = tf.float32)
        
        return image, target
    
class Normalize:
    def __init__(self, mean: list[float], std: list[float]):
        
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("Normalize expects mean and std of length 3 (RGB).")
            
        self._mean = tf.constant(mean, dtype=tf.float32)[None, None, :]
        self._std = tf.constant(std,  dtype=tf.float32)[None, None, :]

    def __call__(self, image, target):
        original_dtype = image.dtype
        
        # Convert image to float32
        image = tf.cast(image, dtype = tf.float32)

        # Scale pixels from [0,255] to [0,1]
        if original_dtype.is_integer:
            image = image / 255.0

        # Applying channel wise normalization
        image = (image - self._mean) / self._std

        return image, target
    
class ClipAndFilterBoxes:
    def __init__(self, min_size: int = 1):
        self._min_size = float(min_size)

    def __call__(self, image: tf.Tensor, target: dict[str, Any]):
        # Clip boxes
        boxes = target['boxes']
        labels = target['labels']

        H, W, _ = tf.shape(image)

        H = tf.cast(H, dtype= tf.float32)
        W = tf.cast(W, dtype= tf.float32)

        boxes = tf.cast(boxes,dtype = tf.float32)

        # Clipping the coordinates
        x1, y1, x2, y2 = tf.split(boxes, num_or_size_splits = 4, axis = -1)

        x1 = tf.clip_by_value(x1, 0.0, W)
        y1 = tf.clip_by_value(y1, 0.0, H)
        x2 = tf.clip_by_value(x2, 0.0, W)
        y2 = tf.clip_by_value(y2, 0.0, H)

        boxes = tf.concat([x1,y1,x2,y2], axis = -1)

        # Checking which of the size is less than min

        w = x2 - x1
        h = y2 - y1

        min_size = tf.constant(self._min_size, dtype = tf.float32)

        valid_width = w >= min_size
        valid_height = h >= min_size

        size_valid = tf.logical_and(valid_width, valid_height)

        size_valid = tf.squeeze(size_valid, axis = -1)

        # Filtering them out
        valid_boxes = tf.boolean_mask(boxes, size_valid)
        valid_labels = tf.boolean_mask(labels, size_valid)


        target['labels'] = valid_labels
        target['boxes'] = valid_boxes


        return image, target

def build_preprocess_pipeline(config: dict[str, Any]):
    preprocesss_opts = config['data'].get('preprocess', {})

    preprocess_config = {
        'input_size': tuple(preprocesss_opts.get('input_size', [224,224])),
        'resize': {
            'mode': preprocesss_opts.get('resize', {}).get('mode', 'stretch'),
            'interp': preprocesss_opts.get('resize', {}).get('interp', 'bilinear'),
        },
        'padding': preprocesss_opts.get('pad', {}).get('value', 0),
        'image': {
            'to_float32': preprocesss_opts.get('image', {}).get('to_float32', True),
            'scale': preprocesss_opts.get('image', {}).get('scale', '0_1'),
        },
        'boxes': {
            'input_format': preprocesss_opts.get('boxes', {}).get('format_in', 'xyxy_pixels'),
            'output_format': preprocesss_opts.get('boxes', {}).get('format_out', 'xyxy_norm'),
            'clip': preprocesss_opts.get('boxes', {}).get('clip', True),
            'min_size': preprocesss_opts.get('boxes', {}).get('min_size', 1),
            'allow_empty': preprocesss_opts.get('boxes', {}).get('allow_empty', True),
            'max_num': preprocesss_opts.get('boxes', {}).get('max_num', None),
        },
        'pipeline': preprocesss_opts.get('standardize_pipeline', ['to_float32', 'scale_01'])
    }

    return preprocess_config
    
def build_augmentation_config(config: dict[str, Any]):
    augment_opts = config['data'].get('augment', {})
    augment_params = augment_opts.get('params', {})

    augment_config = {
        'enabled' : augment_opts.get('enabled', False),
        'output_box_norm': augment_opts.get('output_box_norm', False),
        'pipeline': augment_opts.get('pipeline', ['photometric_distort','random_flip','resize','sanitize_boxes','normalize']),
        'params': {
            'random_flip': {
                'enabled': augment_params.get('random_flip', {}).get('enabled', False),
                'prob': augment_params.get('random_flip', {}).get('prob', 0.1),
                'direction': augment_params.get('random_flip', {}).get('direction', 'horizontal'),
            },
            'random_iou_crop': {
                'enabled': augment_params.get('random_iou_crop', {}).get('enabled', False),
                'prob': augment_params.get('random_iou_crop', {}).get('prob', 0.1),
                'min_iou_choices': augment_params.get('random_iou_crop', {}).get('min_iou_choices', []),
                'min_scale': augment_params.get('random_iou_crop', {}).get('min_scale', 0.3),
                'max_scale': augment_params.get('random_iou_crop', {}).get('max_scale', 1.0),
                'max_attempts': augment_params.get('random_iou_crop', {}).get('max_attempts', 50),
                'fallback': augment_params.get('random_iou_crop', {}).get('fallback', 'original'),
            },
            'random_expand': {
                'enabled': augment_params.get('random_expand', {}).get('enabled', False),
                'prob': augment_params.get('random_expand', {}).get('prob', 0.1),
                'max_ratio': augment_params.get('random_expand', {}).get('max_ratio', 3.0),
                'fill': augment_params.get('random_expand', {}).get('fill', 'mean'),
                'value': augment_params.get('random_expand', {}).get('value', [0.485, 0.456, 0.406]),
            },
            'photometric_distort': {
                'enabled': augment_params.get('photometric_distort', {}).get('enabled', False),
                'prob': augment_params.get('photometric_distort', {}).get('prob', 0.1),
                'brightness': augment_params.get('photometric_distort', {}).get('brightness', 0.125),
                'contrast': augment_params.get('photometric_distort', {}).get('contrast', [0.5, 1.5]),
                'saturation': augment_params.get('photometric_distort', {}).get('saturation', [0.5, 1.5]),
                'hue': augment_params.get('photometric_distort', {}).get('hue', 0.5),
                'random_order': augment_params.get('photometric_distort', {}).get('random_order', True),
            },
            'resize': {
                'enabled': augment_params.get('resize', {}).get('enabled', False),
                'size': augment_params.get('resize', {}).get('size', [300, 300]),
                'mode': augment_params.get('resize', {}).get('mode', 'stretch'),
                'interp': augment_params.get('resize', {}).get('interp', 'bilinear'),
            },
            'sanitize_boxes': {
                'enabled': augment_params.get('sanitize_boxes', {}).get('enabled', False),
                'clip': augment_params.get('sanitize_boxes', {}).get('clip', False),
                'min_size': augment_params.get('sanitize_boxes', {}).get('min_size', 1),
                'min_size_mode': augment_params.get('sanitize_boxes', {}).get('min_size_mode', 'pixels'),
            },
            'normalize': {
                'enabled': augment_params.get('normalize', {}).get('enabled', False),
                'mean': augment_params.get('normalize', {}).get('mean', [0.485, 0.456, 0.406]),
                'std': augment_params.get('normalize', {}).get('std', [0.229, 0.224, 0.225]),
            }
        }
    }

    return augment_config

def build_transforms(config: dict[str, Any]):
    augment_config = build_augmentation_config(config)

    preprocess_config = build_preprocess_pipeline(config)

    # Building the config based on the pipeline
    transform_list = []
    # Iterating over the preprocess config
    for preprocess_transform in preprocess_config['pipeline']:
         match preprocess_transform:
            case 'to_float32':
                float_transform = ToFloat32()
                transform_list.append(float_transform)
            case 'scale_01':
                scale = Scale01()
                transform_list.append(scale)

    # Iterating over the augmentation transforms
    for key in augment_config['pipeline']:
        if not augment_config['params'][key]['enabled']:
            continue

        # Parse the config based on the type
        match key:
            case 'random_flip':
                if augment_config['params'][key]['direction'] == 'horizontal':
                    flip = RandomHorizontalFlip(p = augment_config['params'][key]['prob'])
                else:
                    pass

                transform_list.append(flip)
            case 'random_iou_crop':
                pass
            case 'random_expand':
                pass
            case 'photometric_distort':
                p = augment_config['params'][key]['prob']
                brightness_delta = augment_config['params'][key]['brightness']
                contrast_range = augment_config['params'][key]['contrast']
                saturation_range = augment_config['params'][key]['saturation']
                hue_delta = augment_config['params'][key]['hue']
                channel_swap = augment_config['params'][key]['random_order']
                transform_list.append(PhotometricDistort(p = p, brightness_delta = brightness_delta, contrast_range = contrast_range,saturation_range = saturation_range,hue_delta = hue_delta, channel_swap = channel_swap))
            case 'resize':
                target_size = augment_config['params'][key]['size']
                mode = augment_config['params'][key]['mode']
                transform_list.append(Resize(size = tuple(target_size), mode = mode))
            case 'sanitize_boxes':
                min_size = augment_config['params'][key]['min_size']
                transform_list.append(ClipAndFilterBoxes(min_size = min_size))
            case 'normalize':
                mean = augment_config['params'][key]['mean']
                std = augment_config['params'][key]['std']
                transform_list.append(Normalize(mean = mean, std = std))
            case _:
                raise ValueError("Wrong Transform type present in the config")

    if augment_config['output_box_norm']:
        normalize_boxes = NormalizeBoundingBoxes()
        transform_list.append(normalize_boxes)

    compose = Compose(transforms = transform_list)

    return compose