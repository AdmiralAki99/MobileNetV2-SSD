import tensorflow as tf
from typing import Any, Optional

class OptimizerFactory:
    @staticmethod
    def build(config: dict[str,Any], learning_schedule: tf.keras.optimizers.schedules.LearningRateSchedule | None = None):

        config = OptimizerFactory.build_optimizer_config(config)
        
        name = str(config['name']).strip().lower()
        
        weight_decay = float(config['weight_decay'])
        learning_rate = float(config['lr'])

        learning_schedule = learning_schedule if learning_schedule is not None else learning_rate

        if name == "sgd":
            momentum = float(config['momentum'])
            nesterov = bool(config['nesterov'])
            weight_decay = float(config['weight_decay'])
            return tf.keras.optimizers.SGD(learning_rate = learning_schedule , momentum = momentum, nesterov = nesterov, weight_decay= weight_decay, name = name)
        elif name in ("adam","adamw"):
            beta_1 = float(config['beta1'])
            beta_2 = float(config['beta2'])
            epsilon = float(config['epsilon'])
            if name == "adam":
                return tf.keras.optimizers.Adam(learning_rate = learning_schedule, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, name = name)
            else:
                return tf.keras.optimizers.AdamW(learning_rate = learning_schedule, beta_1 = beta_1, beta_2 = beta_2, epsilon = epsilon, weight_decay = weight_decay, name = name)   
        else:
            raise ValueError(f"Unknown optimizer name: {name}")
                            
    @staticmethod
    def build_optimizer_config(config: dict[str, Any]):
        optimizer_opts = config['optimizer']
        optimizer_config = {
            'name': optimizer_opts.get('name', 'sgd'),
            'lr': optimizer_opts.get('lr', 0.001),
            'weight_decay': optimizer_opts.get('weight_decay', 0.0005),
            'beta1': optimizer_opts.get('beta1', 0.9),
            'beta2': optimizer_opts.get('beta2', 0.999),
            'epsilon': optimizer_opts.get('epsilon', 1e-07),
            'momentum': optimizer_opts.get('momentum', 0.9),
            'nesterov': optimizer_opts.get('nesterov', False),
        }
        return optimizer_config
