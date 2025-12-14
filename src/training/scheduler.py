import tensorflow as tf
from typing import Any

# Need to create a learning rate schedule

class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_learning_rate: float, minimum_learning_rate: float, warmup_steps: int, total_steps:int):
        super(CosineWarmupSchedule, self).__init__()
        
        self.base_learning_rate = tf.constant(base_learning_rate, dtype= tf.float32)
        self.minimum_learning_rate = tf.constant(minimum_learning_rate, dtype= tf.float32)
        self.warmup_steps = tf.constant(warmup_steps, dtype= tf.int64)
        self.total_steps = tf.constant(total_steps, dtype= tf.int64)
        self.pi = tf.constant(3.141592653589793, tf.float32)
        
    def __call__(self, step: tf.Tensor):
        
        step = tf.cast(step, dtype = tf.float32)

        warmup_steps = tf.cast(self.warmup_steps, dtype = tf.float32)
        total_steps = tf.cast(self.total_steps, dtype = tf.float32)

        # Phase 1: Warm up learning rate
        warmup_learning_rate = self.base_learning_rate * tf.minimum(1.0, step / tf.maximum(1.0, warmup_steps))

        # Phase 2: Cosine Decay
        diff_in_steps = tf.maximum(1.0,total_steps - warmup_steps)
        progress = tf.clip_by_value((step - warmup_steps)/diff_in_steps, 0.0, 1.0)
        cosine_decay = self.minimum_learning_rate + 0.5 * (self.base_learning_rate - self.minimum_learning_rate) * (1 + tf.math.cos(self.pi * progress))

        return tf.where(step < warmup_steps, warmup_learning_rate, cosine_decay)       
    
class ConstantLearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_learning_rate: float):
        super(ConstantLearningSchedule, self).__init__()
        
        self.base_learning_rate = tf.constant(base_learning_rate, dtype= tf.float32)
        
    def __call__(self, step: tf.Tensor):
        return self.base_learning_rate      
    
class StepDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_learning_rate: float, gamma: float, milestones: list[int]):
        super(StepDecaySchedule, self).__init__()
        
        self.base_learning_rate = tf.constant(base_learning_rate, dtype= tf.float32)
        self.gamma = tf.constant(gamma, dtype= tf.float32)
        self.milestones = tf.constant(milestones, dtype= tf.int64)
        
    def __call__(self, step: tf.Tensor):
        
        step = tf.cast(step, dtype = tf.int64)

        # Calculating the number of milestones passed
        num_milestones = tf.reduce_sum(tf.cast(step >= self.milestones, dtype= tf.float32))

        # Calculating the learning rate
        learning_rate = self.base_learning_rate * tf.math.pow(self.gamma, num_milestones)

        return learning_rate             
    
class ExponentialDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_learning_rate: float, gamma: float, decay_interval: int):
        super(ExponentialDecaySchedule, self).__init__()
        
        self.base_learning_rate = tf.constant(base_learning_rate, dtype= tf.float32)
        self.gamma = tf.constant(gamma, dtype= tf.float32)
        self.decay_interval = tf.constant(decay_interval, dtype= tf.int64)
        
    def __call__(self, step: tf.Tensor):
        
        step = tf.cast(step, dtype = tf.int64)

        # Calculating the decay factor
        decay_factor = tf.math.pow(self.gamma, tf.cast(step/self.decay_interval, dtype = tf.float32))

        # Calculating the exponential decay
        learning_rate = self.base_learning_rate * decay_factor

        return learning_rate          

# Creating a scheduler

class Scheduler(tf.Module):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, lr_schedule: tf.keras.optimizers.schedules.LearningRateSchedule, start_step: int = 0):
        super().__init__(name="scheduler")
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.current_step = tf.Variable(start_step, dtype=tf.int64, trainable = False)

    @tf.function
    def apply_learning_rate(self):
        learning_rate = self.lr_schedule(self.current_step)
        learning_rate = tf.cast(learning_rate, dtype=tf.float32)
        self.optimizer.learning_rate.assign(learning_rate)
        
        return learning_rate
        
    @tf.function   
    def step(self):
        
        self.current_step.assign_add(1)
        
        learning_rate = self.lr_schedule(self.current_step)
        
        self.optimizer.learning_rate.assign(learning_rate)
        
        return learning_rate

def create_scheduler_config_from_main_config(config: dict[str,Any]):
    scheduler = config['train'].get('scheduler')
    warmup_config = scheduler.get('warmup', {})
    step_config = scheduler.get('step', {})
    multistep_config = scheduler.get('multistep', {})
    
    schedule_config = {
        'learning_schedule': scheduler.get('name',"constant"),
        'base_learning_rate': scheduler.get('base_lr', 0.1),
        'mininum_learning_rate': scheduler.get('min_lr', 0.001),
        'total_steps': scheduler.get('total_steps', None),
        'warmup_epochs': warmup_config.get('epochs',10),
        'warmup_steps': warmup_config.get('steps', 15),
        'warmup_enabled': warmup_config.get('enabled',True),
        'warmup_start_factor': warmup_config.get('start_factor',0.1),
        'warmup_end_factor': warmup_config.get('end_factor',0.1),
        'warmup_mode': warmup_config.get('mode',0.1),
        'step_drop_every_epochs': step_config.get('drop_every_epochs', None),
        'step_gamma': step_config.get('gamma', None),
        'multistep_milestones': multistep_config.get('milestones_epochs', [])
    }
    return schedule_config

def build_learning_schedule(config: dict[str,Any]):
    # Need to select the correct learning schedule
    if config['learning_schedule'] == 'cosine_warmup':
        learning_schedule = CosineWarmupSchedule(base_learning_rate = config['base_learning_rate'], minimum_learning_rate = config['mininum_learning_rate'], warmup_steps = config['warmup_steps'], total_steps = config['total_steps'])
    elif config['learning_schedule'] == 'step_decay':
        learning_schedule = StepDecaySchedule(base_learning_rate = config['base_learning_rate'], gamma = config['step_gamma'], milestones = config['multistep_milestones'])
    elif config['learning_schedule'] == 'exponential_decay':
        learning_schedule = ExponentialDecaySchedule(base_learning_rate = config['base_learning_rate'], gamma = config['step_gamma'], decay_interval = config['step_drop_every_epochs'])
    else:
        learning_schedule = ConstantLearningSchedule(base_learning_rate = config['base_learning_rate'])

    return learning_schedule

def build_scheduler(config: dict[str,Any], optimizer: tf.keras.optimizers):
    
    scheduler_config = create_scheduler_config_from_main_config(config)

    learning_schedule = build_learning_schedule(scheduler_config)

    scheduler = Scheduler(optimizer = optimizer, lr_schedule = learning_schedule, start_step = 10)

    return scheduler