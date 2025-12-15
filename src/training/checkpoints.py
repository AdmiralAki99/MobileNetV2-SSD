from dataclasses import dataclass
import tensorflow as tf
from typing import Any, Optional
from pathlib import Path
import hashlib
import json

from .scheduler import Scheduler

class CheckpointManager:
    def __init__(self, checkpoint_config: dict[str,Any], model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, scheduler: Scheduler, ema: Optional[Any], is_main_node: bool = True):
        self._model = model
        self._optimizer = optimizer
        
        self._checkpoint_directory = Path(checkpoint_config['dir'])
        self._checkpoint_directory.mkdir(parents = True, exist_ok = True)
        
        self._keep_last_k = checkpoint_config.get('keep_last_k', 5)
        
        self._save_every_steps = checkpoint_config.get('save_every_steps', None)
        if self._save_every_steps is not None and self._save_every_steps <= 0:
            raise ValueError("The value has to be greater than 0")
        
        self._save_every_epochs = checkpoint_config.get('save_every_epochs', 1)
        self._save_last = checkpoint_config.get('save_last', True)
        self._save_best = checkpoint_config.get('save_best', True)
        self._monitor = checkpoint_config.get('monitor', "val_map")
        self._mode = checkpoint_config.get('mode', "max")

        if self._mode not in {"max","min"}:
            raise ValueError("The value for the mode is wrong and should be either 'max' or 'min'")
        
        self._ema = ema
        self._sched = scheduler

        # Saving the main status (rank = 0) to stop potential I/O problems when using DDP
        self._is_main = is_main_node

        # Creating the variables to use
        self._epoch_var = tf.Variable(0, dtype = tf.int64, trainable = False)
        self._global_step_var = tf.Variable(0, dtype = tf.int64, trainable = False)
        self._best_epoch_var = tf.Variable(-1, dtype = tf.int64, trainable = False)
        self._best_metric_var = tf.Variable(float("-inf"), dtype = tf.float32, trainable = False) if self._mode == "max" else tf.Variable(float("inf"), dtype = tf.float32, trainable = False)

        # Building the checkpoint bundle for the manager to store
        checkpoint_dict = {
            'model': self._model,
            'scheduler': self._sched,
            'epoch': self._epoch_var,
            'global_step': self._global_step_var,
            'best_epoch': self._best_epoch_var,
            'best_metric': self._best_metric_var
        }

        if self._ema is not None:
            checkpoint_dict['ema'] = self._ema
        
        self._checkpoint = tf.train.Checkpoint(**checkpoint_dict)

        # Ensuring the /last subdirectory is creating in the checkpoint 
        self._last_directory = self._checkpoint_directory / "last"
        self._last_directory.mkdir(parents = True, exist_ok = True)

        # Now creating the two checkpoint managers that will be used (last & best)
        self._last_manager = tf.train.CheckpointManager(checkpoint = self._checkpoint, directory = str( self._last_directory), max_to_keep = self._keep_last_k)
        self._best_manager = None

        # This manager is used when there is a metric increase.
        if self._save_best:
            # There is a save best manager that is used
            self._best_directory = self._checkpoint_directory / "best"
            self._best_directory.mkdir(parents = True, exist_ok = True)
            self._best_manager = tf.train.CheckpointManager(checkpoint = self._checkpoint, directory = str(self._best_directory), max_to_keep = 1)
        
    def restore_latest(self):
        # Accessing the last manager and its parts
        latest_path = self._last_manager.latest_checkpoint

        if latest_path is None:
            latest_dir = Path(self._last_manager.directory)
            index_files = list(latest_dir.glob("ckpt-*.index"))
            if index_files:
                newest = max(index_files, key = self._select_checkpoint)
                latest_path = str(newest.with_suffix(""))

        # Checking if the path is None
        if latest_path is None:
            return {'restored': False, 'epoch': 0, 'global_step': 0, 'best_metric': float("-inf") if self._mode == "max" else float("inf") , 'best_epoch': -1}

        # There is a checkpoint and now needs to be loaded
        self._checkpoint.restore(latest_path).expect_partial()

        # Now getting the values to return to the training loop to resume correctly
        epoch = int(self._epoch_var.numpy())
        global_step = int(self._global_step_var.numpy())
        best_metric = float(self._best_metric_var.numpy())
        best_epoch = int(self._best_epoch_var.numpy())

        return {'restored': True, 'epoch': epoch, 'global_step': global_step, 'best_metric': best_metric , 'best_epoch': best_epoch}

    def restore_best(self):
        # Check if the best manager is even initialized since it can be disabled
        if self._best_manager is None:
            return {'restored': False, 'epoch': 0, 'global_step': 0, 'best_metric': float("-inf") if self._mode == "max" else float("inf") , 'best_epoch': -1}

        # Now check if the checkpoint exists
        best_path = self._best_manager.latest_checkpoint

        if best_path is None:
            best_dir = Path(self._best_manager.directory)
            index_files = list(best_dir.glob("ckpt-*.index"))
            if index_files:
                newest = max(index_files, key = self._select_checkpoint)
                best_path = str(newest.with_suffix(""))
                
        # Check if the path exists
        if best_path is None:
            return {'restored': False, 'epoch': 0, 'global_step': 0, 'best_metric': float("-inf") if self._mode == "max" else float("inf") , 'best_epoch': -1}

        # The path exists and now needs to be restored
        self._checkpoint.restore(best_path).expect_partial()

        # Now getting the values to return to the training loop to resume correctly
        epoch = int(self._epoch_var.numpy())
        global_step = int(self._global_step_var.numpy())
        best_metric = float(self._best_metric_var.numpy())
        best_epoch = int(self._best_epoch_var.numpy())

        return {'restored': True, 'epoch': epoch, 'global_step': global_step, 'best_metric': best_metric , 'best_epoch': best_epoch}

    def save_last(self, epoch: int, global_step: int):

        if not self._is_main:
            return None

        # Now saving the variables
        self._epoch_var.assign(epoch)
        self._global_step_var.assign(global_step)

        # Path create checkpoint file name
        save_path = self._last_manager.save(checkpoint_number = global_step)

        return save_path
        
    def save_best(self, epoch: int, global_step: int, metric: float):
        if not self._is_main:
            return {'is_best': False, 'path': None}
        
        if not self._save_best or self._best_manager is None:
            return {'is_best': False, 'path': None}

        # Now checking if the metric is less than or more than the value
        if self._compare_metrics(metric):
            
            # Assign the metric and the epoch to track the best
            self._best_metric_var.assign(metric)
            self._best_epoch_var.assign(epoch)
            self._epoch_var.assign(epoch)
            self._global_step_var.assign(global_step)

            # Save the checkpoint
            best_path = self._best_manager.save(checkpoint_number = global_step)

            return {'is_best': True, 'path': best_path}

        return {'is_best': False, 'path': None}

    def _compare_metrics(self, metric: float):

        # Now comparing the metric
        if self._mode == "max":
            return metric > float(self._best_metric_var.numpy())
        else:
            return metric < float(self._best_metric_var.numpy())

    def _select_checkpoint(self,p: Path):
        # "ckpt-123.index" -> 123
        stem = p.name.split(".")[0]      # "ckpt-123"
        return int(stem.split("-")[1])   # 123

    def should_save_step(self, global_step: int):
        # Now checking if the conditions for the steps are not violated
        if not self._save_last:
            return False

        # Checking if the checkpoint needs to saved every k steps
        if self._save_every_steps is None:
            return False

        # Now checking if the step is exactly the interval to save the checkpoint on
        if global_step % self._save_every_steps == 0:
            return True

        return False

    def should_save_epoch(self,epoch: int):
        # Now checking if the conditions for the epochs are not violated
        if not self._save_last:
            return False

        # Default is to save every epoch so there is no need for the second condition
        if (epoch + 1) % self._save_every_epochs == 0:
            return True

        return False
    
def _create_checkpoint_directory_fingerprint(config: dict[str,Any]):
    model_config = config['model']
    dataset_config = config['data']
    train_config = config['train']

    fingerprint_config = {
        'model_backbone': model_config.get('backbone',''),
        'num_classes': model_config.get('num_classes',0),
        'priors': model_config.get('priors',{}),
        'dataset_name': dataset_config.get('dataset_name', ''),
        'dataset_augmentation': dataset_config.get('augment', {}),
        'dataset_normalization': dataset_config.get('normalization', {}),
        'training_batch_size': train_config.get('batch_size', 0),
        'training_epochs': train_config.get('epochs', 0),
        'training_optimizer_name': train_config['optimizer'].get('name', ''),
        'training_optimizer_lr': train_config['optimizer'].get('lr', 0.0),
        'training_optimizer_weight_decay': train_config['optimizer'].get('weight_decay', 0.0),
        'training_scheduler_params':train_config.get('scheduler',{})
    }

    config_json = json.dumps(fingerprint_config,sort_keys = True, separators=(",",":"))
    hash_object = hashlib.sha256(config_json.encode('utf-8'))
    hex_digest = hash_object.hexdigest()[:10]

    file_slug = f"{model_config['name']}_{dataset_config['dataset_name']}_img{dataset_config['input_size'][0]}_bs{fingerprint_config['training_batch_size']}_lr{fingerprint_config['training_optimizer_lr']:.2e}_{train_config['scheduler']['name']}"
    file_name = f"{file_slug}_{hex_digest}"

    root_dir =  config['checkpoint']['dir']
    run_dir = Path(root_dir) / file_name
    
    return {
        'dir': str(run_dir),
        'keep_last_k': config['checkpoint'].get('keep_last_k', 1),
        'save_every_steps': config['checkpoint'].get('save_every_steps', 200),
        'save_every_epochs': config['checkpoint'].get('save_every_epochs', 1),
        'save_last': config['checkpoint'].get('save_last', True),
        'save_best': config['checkpoint'].get('save_best', True),
        'monitor': config['checkpoint'].get('monitor', 'val_map'),
        'mode': config['checkpoint'].get('mode', 'max')
    }

def build_checkpoint_manager(config: dict[str,Any], model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, scheduler: Scheduler, ema: Optional[Any], is_main_node: bool =True):
    checkpoint_config = _create_checkpoint_directory_fingerprint(config)

    checkpoint_manager = CheckpointManager(checkpoint_config, model, optimizer, scheduler, ema= ema, is_main_node= is_main_node)

    return checkpoint_manager
    