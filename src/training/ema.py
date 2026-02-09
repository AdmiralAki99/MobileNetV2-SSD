import tensorflow as tf
from typing import Any, Optional
from contextlib import contextmanager

class EMA(tf.Module):
    def __init__(self, model: tf.keras.Model, ema_config: dict[str, Any]):
        super().__init__(name="EMA")

        if model is None:
            raise ValueError("EMA requires a built model instance (model cannot be None).")
        if not model.built:
            raise ValueError("Build/call the model before creating EMA, otherwise trainable_variables is empty.")
        
        self._decay = float(ema_config.get('decay', 0.999))
        self._enabled = bool(ema_config.get('enabled', True))
        
        self._warmup_steps = int(ema_config.get('warmup_steps', 0))
        self._update_every = int(ema_config.get('update_every', 1))
        self._eval_use_ema = bool(ema_config.get('eval_use_ema', True))
        self._is_applied = False
        
        self._num_updates = tf.Variable(0, dtype=tf.int64, trainable=False, name="num_updates") # Tracking the counter
        self._use_num_updates = bool(ema_config.get('use_num_updates', False))

        # Need to initialize the model training variables
        self._model_vars = list(model.trainable_variables)

        self._ema_vars = [tf.Variable(tf.convert_to_tensor(variable), dtype = variable.dtype, trainable = False, name = f"{variable.name.replace(':', '_')}_ema") for variable in self._model_vars]

        self._backup = None

    def reset(self):
        # The function needs to reset to the models current weights

        # First check if the model weights and EMA weights are mapped 1:1
        if len(self._ema_vars) != len(self._model_vars):
            raise ValueError("EMA values are not 1:1 check the length of the variables passed to the EMA.")

        # Need to copy the current weights of the model into the EMA
        for ema_var, model_var in zip(self._ema_vars,self._model_vars):
            ema_var.assign(model_var)

        # Need to reset the updates since the EMA was reset to the model's weights
        self._num_updates.assign(0)

        # Clearing the cache of values
        self._backup = None

    def should_update(self, step: int):

        # Checking if the step is in the warmup phase or not
        if step < self._warmup_steps:
            return False

        # Checking if the step is between the range acceptable
        if self._update_every > 1 and step % self._update_every != 0:
            return False

        # Checking if EMA is enabled
        if not self._enabled:
            return False

        return True # Everything passed the conditions

    def should_apply_during_eval(self):
        return self._enabled and self._eval_use_ema and len(self._ema_vars) > 0

    def update(self, step: int):
        # Function updates the value of the EMA

        if not self.should_update(step):
            return

        decay = tf.constant(self._decay, tf.float32)
        num_updates = tf.cast(self._num_updates, tf.float32)

        # Calculating the ramp based on how may updates have been made to account for early garbage weights
        adjusted_decay = (1 + num_updates)/ (10 + num_updates)

        # Selecting the minimum of the two
        decay_rate = tf.minimum(decay, adjusted_decay)
        decay_rate = tf.cast(decay_rate, tf.float32)
        
        inverse_decay_rate = 1 - decay_rate
        
        # Now updating the value
        for ema_var, model_var in zip(self._ema_vars,self._model_vars, strict = True):
            decay_factor = tf.cast(decay_rate, ema_var.dtype)
            inverse_decay_factor = tf.cast(1.0, ema_var.dtype) - decay_factor
            ema_var.assign(decay_factor * ema_var + inverse_decay_factor * model_var)

        # Increment the counter
        self._num_updates.assign_add(1)

    def apply_to(self, model: tf.keras.Model | None = None):
        # Need to check if the model is None to pick the correct one
        if model is None:
            # Using the fallback model
            model_variables = self._model_vars
        else:
            model_variables = list(model.trainable_variables)
            
        if len(self._ema_vars) != len(model_variables):
            raise ValueError("EMA vars and model vars mismatch")

        # Checking if the dtype is correct
        for ema_var, model_var in zip(self._ema_vars, model_variables):
            if ema_var.dtype != model_var.dtype or ema_var.shape != model_var.shape:
                raise ValueError("Dtypes not same for target model and EMA saved copy")

        # Now checking if backup exists so if used consecutively there can be a sort of ECF with the EMA weights
        if (self._backup is not None) or (self._is_applied):
            raise ValueError("Cannot apply since backup exists, restore() needs to be called")

        # Creating a backup
        self._backup = [tf.convert_to_tensor(var) for var in model_variables]

        # Now swapping the ema weights into the model weights
        for ema_var, model_var in zip(self._ema_vars,model_variables):
            model_var.assign(ema_var)

        self._is_applied = True

    def restore(self, model: tf.keras.Model | None = None):
        # Need to check if the model is None to pick the correct one
        if model is None:
            # Using the fallback model
            model_variables = self._model_vars
        else:
            model_variables = list(model.trainable_variables)

        # Checking if there is a backup to restore from
        if (self._backup is None) or (not self._is_applied):
            raise ValueError("Cannot restore since backup doesnt exist, apply_to() needs to be called")

        if len(self._backup) != len(model_variables):
            raise ValueError("Backup and model vars mismatch")

        # Restoring the backup to the model
        for backup_var, model_var in zip(self._backup,model_variables):
            if backup_var.dtype != model_var.dtype or backup_var.shape != model_var.shape:
                raise ValueError("Dtypes not same for target model and Backup copy")
            model_var.assign(backup_var)

        # Clearing the backup
        self._backup = None
        self._is_applied = False
    
    @contextmanager
    def eval_context(self, model: tf.keras.Model | None = None):
        # This function needs to check whether the model requires EMA or not for the evaluation lifecycle
        use_ema = self.should_apply_during_eval()

        if use_ema:
            # Apply the ema weights to the model
            self.apply_to(model)

        try:
            yield # Allow for the eval step to run without an issue
        finally:
            # If ema was then it needs to be reverted to the model raw values for the training step once again
            if use_ema:
                self.restore(model)
                
def get_ema_config(config: dict[str,Any]):
    ema_options= config.get('ema',{})

    ema_config = {
        'enabled': ema_options.get('enabled',True),
        'decay': ema_options.get('decay', 0.9),
        'warmup_steps': ema_options.get('warmup_steps', 0),
        'update_every': ema_options.get('update_every', 1),
        'eval_use_ema': ema_options.get('eval_use_ema', True)
    }

    return ema_config

def build_ema(config: dict[str,Any], model: tf.keras.Model):
    # First build the config
    ema_config = get_ema_config(config)

    # Build the EMA
    ema = EMA(model = model, ema_config = ema_config)

    return ema