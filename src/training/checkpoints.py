import tensorflow as tf
from typing import Any, Optional
from pathlib import Path
from mobilenetv2ssd.core.fingerprint import Fingerprint
from datetime import datetime, timezone

class CheckpointManager:
    def __init__(self, checkpoint_config: dict[str,Any], model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, ema: Optional[Any], is_main_node: bool = True):
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
            'epoch': self._epoch_var,
            'global_step': self._global_step_var,
            'best_epoch': self._best_epoch_var,
            'best_metric': self._best_metric_var
        }
        
        # Adding optimizers to the checkpoint
        checkpoint_dict[f'optimizer'] = self._optimizer

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
        
    @property
    def last_directory(self):
        return self._last_directory

    @property
    def best_directory(self):
        return self._best_directory

    @property
    def log_directory(self):
        return self._checkpoint_directory.parent

    def build_optimizer(self, var_group: list[tf.Variable]):
        # Need to build the singular optimizer 
        if isinstance(self._optimizer, tf.keras.optimizers.Optimizer):
            self._optimizer.build(var_group)
    
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

    def restore_from_directory(self, external_checkpoint_path: Path):
        # Need to check if the path exists
        if not external_checkpoint_path.exists():
            return {'restored': False, 'epoch': 0, 'global_step': 0, 'best_metric': float("-inf") if self._mode == "max" else float("inf") , 'best_epoch': -1}
    
        # The path exists and now can be parsed
        self._checkpoint.restore(str(external_checkpoint_path)).expect_partial()
        
        restored_epoch = int(self._epoch_var.numpy())
        restored_global_step = int(self._global_step_var.numpy())
        restored_best_metric = float(self._best_metric_var.numpy())
        restored_best_epoch = float(self._best_epoch_var.numpy())
        
        return {'restored': True, 'epoch': restored_epoch, 'global_step': restored_global_step, 'best_metric': restored_best_metric , 'best_epoch': restored_best_epoch}
    
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
    
    def _normalize_optimizers(self, optimizer: tf.keras.optimizers.Optimizer | dict[str, tf.keras.optimizers.Optimizer]):
        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            return {'main': optimizer}
        
        # There are multiple optimizers present
        if isinstance(optimizer, dict):
            if not optimizer:
                raise ValueError("The optimizer dictionary is empty")
            
            for key, opt in optimizer.items():
                # Checking if the instance is correct
                if not isinstance(key, str):
                    raise ValueError("The optimizer dictionary keys must be strings")
                
                if not isinstance(opt, tf.keras.optimizers.Optimizer):
                    raise TypeError(f"optimizer['{key}'] is not a tf.keras.optimizers.Optimizer")
                
            return dict(optimizer)
        
        raise TypeError("The optimizer must be either a tf.keras.optimizers.Optimizer or a dictionary of them")

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
    
def _create_checkpoint_directory_fingerprint(config: dict[str,Any], fingerprint: Fingerprint = None, job_dir: Path | str = None):
    
    experiment_name = config.get("experiment", {}).get("id", "default_experiment")
    fingerprint_str = str(fingerprint.short) if fingerprint else "no_fingerprint"
    name_format = config.get("run", {}).get('name_format', "{experiment_id}_{fingerprint}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name_format = name_format.replace("{experiment_id}", experiment_name)
    name_format = name_format.replace("{fingerprint}", fingerprint_str)
    
    job_name = name_format
    run_dir = config.get("run", {}).get("root", "runs")
    log_dir = config.get("run", {}).get("subdirs", {}).get('logs', 'logs')
    checkpoint_dir = config.get("run", {}).get("subdirs", {}).get('checkpoints', 'checkpoints')
    
    run_dir = Path(run_dir)
    if job_dir is None:
        checkpoint_dir = run_dir / job_name / log_dir / timestamp / checkpoint_dir
    else:
        if isinstance(job_dir, str):
            job_dir = Path(job_dir)
            
        checkpoint_dir = Path(job_dir) / checkpoint_dir
    
    return {
        'dir': str(checkpoint_dir),
        'keep_last_k': config['checkpoint'].get('keep_last_k', 1),
        'save_every_steps': config['checkpoint'].get('save_every_steps', 200),
        'save_every_epochs': config['checkpoint'].get('save_every_epochs', 1),
        'save_last': config['checkpoint'].get('save_last', True),
        'save_best': config['checkpoint'].get('save_best', True),
        'monitor': config['checkpoint'].get('monitor', 'val_map'),
        'mode': config['checkpoint'].get('mode', 'max')
    }

def build_checkpoint_manager(config: dict[str,Any], model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer | dict[str, tf.keras.optimizers.Optimizer], ema: Optional[Any], is_main_node: bool =True, fingerprint: Fingerprint= None, log_dir: Path | None = None):
    
    checkpoint_config = _create_checkpoint_directory_fingerprint(config, fingerprint= None if fingerprint is None else fingerprint, job_dir= log_dir)

    checkpoint_manager = CheckpointManager(checkpoint_config, model, optimizer=optimizer, ema= ema, is_main_node= is_main_node)

    return checkpoint_manager