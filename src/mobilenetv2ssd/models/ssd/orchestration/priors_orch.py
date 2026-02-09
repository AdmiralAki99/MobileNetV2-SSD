import tensorflow as tf
import hashlib, json
from typing import Any

from mobilenetv2ssd.core.fingerprint import Fingerprinter
from mobilenetv2ssd.models.ssd.ops.anchor_ops_tf import build_priors, build_priors_batched

def _extract_information_from_model_config(model_config : dict[str, Any]):
    config = model_config['priors']
    prior_config = {
        # Big prior hyperparameters
        "image_size": config['image_size'],
        "strides": config['strides'],
        "feature_map_shapes": None if 'feature_map_shapes' not in config else config['feature_map_shapes'],
        
        # Prior Shape Determinants
        "min_scale": config['min_scale'],
        "max_scale": config['max_scale'],
        "scales": None if 'scales' not in config else config['scales'],
        "aspect_ratios": config['aspect_ratios'],

        # Extra options that can be added in the model
        "two_scales_per_octave": True, # Saw this in a article about RetinaNet and just added the option for later iterations
        "extra_scales_per_layer": True,
        "format": "cxcywh",
        "normalize": True, # Always assumes normalization but can be added in future iterations for more control
        "clip": True,
        "dtype": "float32", # Important for later since I will be using this to shrink the computation on embedded hardware

        # Tilting
        "center_offset" : 0.5,
        "align_corners" : False,
    }

    return prior_config

def _compute_prior_config_fingerprint(config):
    return Fingerprinter().fingerprint(config).hex

def _validate_prior_config(config):
    # Checking if the format of the config is correct
    image_size = config.get("image_size")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("image_size must be a sequence of length 2 (H, W).")

    h, w = image_size
    if not (isinstance(h, (int, float)) and isinstance(w, (int, float))):
        raise ValueError("image_size values must be numeric.")
    if not (h > 0 and w > 0):
        raise ValueError("image_size must be positive in both dimensions.")

    # Checking if the length of the strides and the aspect_ratios is the same
    strides = config.get("strides")
    fm_shapes = config.get("feature_map_shapes")

    if strides is None and fm_shapes is None:
        raise ValueError("At least one of 'strides' or 'feature_map_shapes' must be provided.")

    num_levels = None

    if strides is not None:
        if not isinstance(strides, (list, tuple)) or len(strides) == 0:
            raise ValueError("'strides' must be a non-empty list when provided.")
        for s in strides:
            if not (isinstance(s, int) and s > 0):
                raise ValueError("All strides must be positive integers.")
        num_levels = len(strides)

    if fm_shapes is not None:
        if not isinstance(fm_shapes, (list, tuple)) or len(fm_shapes) == 0:
            raise ValueError("'feature_map_shapes' must be a non-empty list when provided.")
        for shape in fm_shapes:
            if not isinstance(shape, (list, tuple)) or len(shape) != 2:
                raise ValueError(
                    "Each entry in 'feature_map_shapes' must be a (h, w) pair."
                )
            h_l, w_l = shape
            if not (isinstance(h_l, int) and isinstance(w_l, int)):
                raise ValueError(
                    "Entries in 'feature_map_shapes' must be integer (h, w) pairs."
                )
            if not (h_l > 0 and w_l > 0):
                raise ValueError(
                    "Each feature map shape (h, w) must be positive."
                )

        if num_levels is None:
            num_levels = len(fm_shapes)
        else:
            if len(fm_shapes) != num_levels:
                raise ValueError(
                    "Length of 'feature_map_shapes' must match length of 'strides' "
                    f"({len(fm_shapes)} vs {num_levels})."
                )

    if num_levels is None:
        raise ValueError("Internal error: num_levels could not be inferred from config.")

    aspect_ratios = config.get("aspect_ratios")

    if aspect_ratios is not None:
        if not isinstance(aspect_ratios, (list, tuple)) or len(aspect_ratios) == 0:
            raise ValueError("'aspect_ratios' must be a non-empty list when provided.")

        first_ar = aspect_ratios[0]

        # Helper inline checks (no inner functions)
        if isinstance(first_ar, (int, float)):
            # 1D: broadcast later inside standardize_aspect_ratios
            for ar in aspect_ratios:
                if not isinstance(ar, (int, float)):
                    raise ValueError("All aspect ratio values must be numeric.")
                if not (ar > 0):
                    raise ValueError("All aspect ratio values must be positive.")
        else:
            # 2D: per level
            if len(aspect_ratios) != num_levels:
                raise ValueError(
                    "Length of 'aspect_ratios' (per-level) must match num_levels "
                    f"({len(aspect_ratios)} vs {num_levels})."
                )
            for lvl_idx, lvl_ars in enumerate(aspect_ratios):
                if not isinstance(lvl_ars, (list, tuple)) or len(lvl_ars) == 0:
                    raise ValueError(
                        f"'aspect_ratios[{lvl_idx}]' must be a non-empty list of numbers."
                    )
                for ar in lvl_ars:
                    if not isinstance(ar, (int, float)):
                        raise ValueError(
                            f"All aspect ratios in 'aspect_ratios[{lvl_idx}]' "
                            "must be numeric."
                        )
                    if not (ar > 0):
                        raise ValueError(
                            f"All aspect ratios in 'aspect_ratios[{lvl_idx}]' "
                            "must be positive."
                        )
            

    scales = config.get("scales")
    min_scale = config.get("min_scale")
    max_scale = config.get("max_scale")

    if scales is not None:
        if not isinstance(scales, (list, tuple)) or len(scales) == 0:
            raise ValueError("'scales' must be a non-empty list when provided.")

        first_scale = scales[0]

        if isinstance(first_scale, (int, float)):
            # 1D list of scales
            for s in scales:
                if not isinstance(s, (int, float)):
                    raise ValueError("All scale values in 'scales' must be numeric.")
                if not (0 < s <= 1):
                    raise ValueError(
                        f"Scale value {s} in 'scales' is out of range; "
                        "must satisfy 0 < s <= 1."
                    )
        else:
            # 2D: per level
            if len(scales) != num_levels:
                raise ValueError(
                    "Length of 'scales' (per-level) must match num_levels "
                    f"({len(scales)} vs {num_levels})."
                )
            for lvl_idx, lvl_scales in enumerate(scales):
                if not isinstance(lvl_scales, (list, tuple)) or len(lvl_scales) == 0:
                    raise ValueError(
                        f"'scales[{lvl_idx}]' must be a non-empty list of numbers."
                    )
                for s in lvl_scales:
                    if not isinstance(s, (int, float)):
                        raise ValueError(
                            f"All scale values in 'scales[{lvl_idx}]' must be numeric."
                        )
                    if not (0 < s <= 1):
                        raise ValueError(
                            f"Scale value {s} in 'scales[{lvl_idx}]' is out of range; "
                            "must satisfy 0 < s <= 1."
                        )

        # If explicit scales are provided, min_scale/max_scale are optional.
        # You can optionally add extra consistency checks here if you want.
    else:
        # No explicit scales → we must have valid min_scale + max_scale
        if min_scale is None or max_scale is None:
            raise ValueError(
                "When 'scales' is None, both 'min_scale' and 'max_scale' must be provided."
            )
        if not isinstance(min_scale, (int, float)) or not isinstance(max_scale, (int, float)):
            raise ValueError("'min_scale' and 'max_scale' must be numeric when provided.")
        if not (0 < min_scale <= max_scale <= 1):
            raise ValueError(
                "The relationship 0 < min_scale <= max_scale <= 1 must hold "
                f"(got min_scale={min_scale}, max_scale={max_scale})."
            )
            
def _convert_dtype_to_tf(dtype: str):
    dtype_converter = {
        'int32': tf.int32,
        'int16': tf.int16,
        'int64': tf.int64,
        'int8': tf.int8,
        'float16': tf.float16,
        'float32': tf.float32,
        'float64': tf.float64,        
    }

    return dtype_converter.get(dtype, tf.float32)

def _cache_priors(fingerprint: str, priors, meta: dict):
    pass

def _get_cached_priors(fingerprint: str):
    return None

def build_priors_from_config(model_config,batch_size: int| None = None, evaluation_config = None):
    # The function should be doing very simple steps on top of the operations
    # Steps:
    # 1. Extract the configuration used to create the priors from model_config
    # 2. Compute a config hash (Later implement a small cache system to reduce computations)
    # 3. Validate if the config is correct
    # 3. Computer the priors for one image
    # 4. Batch those priors to be used for all the images (kept as is and then the model refines it using deltas)
    
    prior_config = _extract_information_from_model_config(model_config)
    prior_config['fingerprint'] = _compute_prior_config_fingerprint(prior_config)
    
    _validate_prior_config(prior_config)
    
    # Check if the config exists in the model
    cached = _get_cached_priors(prior_config['fingerprint'])
    if cached is not None:
        priors, meta = cached
    else:
        priors,meta = build_priors(image_size = prior_config['image_size'], strides = prior_config['strides'], feature_map_shapes = prior_config['feature_map_shapes'],scales = prior_config['scales'],aspect_ratios = prior_config['aspect_ratios'],s_min = prior_config['min_scale'],s_max = prior_config['max_scale'],include_extra = prior_config['extra_scales_per_layer'],clip = prior_config['clip'],dtype= _convert_dtype_to_tf(prior_config['dtype']))
        # Cache Priors
        _cache_priors(meta['fingerprint'],priors,meta)

    if batch_size is not None:
        priors = build_priors_batched(priors,batch_size)

    return priors, meta