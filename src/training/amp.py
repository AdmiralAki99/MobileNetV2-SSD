import tensorflow as tf
from typing import Any
from contextlib import contextmanager

from mobilenetv2ssd.core.precision_config import PrecisionConfig

class AMPContext:
    def __init__(self, config: dict[str, Any], optimizer: tf.keras.optimizers.Optimizer):
        self._enabled = config['enabled']
        self._policy = config['policy']
        self._loss_scale : str | float = config['loss_scale']
        self._clip_unscaled_grads = config['clip_unscaled_grads']
        self._force_fp32 = set(config['force_fp32'])
        self._policy_set = False
        self._base_optimizer = optimizer
        self.optimizer = optimizer # Use this optimzer only to handle the mixed precision

    def setup_policy(self):

        # Guarding against different strings
        if self._policy not in {'mixed_float16', 'mixed_bfloat16', 'float32'}:
            raise ValueError(f"AMP policy name is not valid, error value: {self._policy}, allowed values: {['mixed_float16', 'mixed_bfloat16', 'float32']}")
        
        # Check if the policy is set
        if self._policy_set:
            return
        
        # Need to check if amp is enabled or not
        if self._enabled:
            # Enable the global precision policy
            policy = tf.keras.mixed_precision.Policy(self._policy)
            tf.keras.mixed_precision.set_global_policy(policy)
            self._policy_set = True
            return
        else:
            # Setting it to float32 policy even though it is default behaviour
            policy = tf.keras.mixed_precision.Policy("float32")
            tf.keras.mixed_precision.set_global_policy(policy)
            self._policy_set = True
            return

    def wrap_optimizer(self):
        # Need to check if AMP is even on
        if not self._enabled:
            # AMP is off so return the optimzer with the base version
            self.optimizer = self._base_optimizer
            return self.optimizer

        # Now worrying about the loss scaling mode since AMP is on
        if self._loss_scale == "dynamic":
            
            # Wrapping the optimizer in the loss scale mode to allow for mixed precision
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self._base_optimizer)

        else:
            # It is a number and needs to be positive
            if isinstance(self._loss_scale, (int,float)) and self._loss_scale > 0:
                loss_scale_mode = float(self._loss_scale)
            else:
                raise ValueError("Loss Scale is invalid, needs to be 'dynamic' or a positive int or float")

            # TODO: Add support later by going through the documentation for fixed scaling
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self._base_optimizer, initial_scale = loss_scale_mode)
        
        return self.optimizer

    def scale_loss(self, gradients):
        # Keeping this here for later if I decide to take control of the scaling and the clipping of the values
        return gradients
    
    @contextmanager
    def autocast(self):
        yield

    def state_metadata(self):
        return {
            'enabled': self._enabled,
            'policy': self._policy,
            'loss_scale': self._loss_scale,
            'clip_unscaled_grads': self._clip_unscaled_grads,
            'force_fp32': self._force_fp32
        }

    def make_precision_config(self):
        return PrecisionConfig(self._force_fp32)
    
    
def build_amp_config(config: dict[str, Any]):
    train_config = config['train']
    amp_opts = train_config.get('amp',{})

    amp_config = {
        'enabled': amp_opts.get('enabled', False),
        'policy' : amp_opts.get('policy', "float32"),
        'loss_scale': amp_opts.get('loss_scale', 'dynamic'),
        'clip_unscaled_grads' : amp_opts.get('clip_unscaled_grads', True),
        'force_fp32': amp_opts.get('force_fp32', {}),
    }

    return amp_config

def build_amp(config: dict[str, Any], optimizer: tf.keras.optimizers.Optimizer):
    # Build the AMP config
    amp_config = build_amp_config(config)

    # Build the AMP
    amp = AMPContext(amp_config, optimizer)

    return amp