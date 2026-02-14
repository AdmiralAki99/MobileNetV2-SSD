from pathlib import Path
import traceback
import sys
from typing import Any
from datetime import datetime, timezone
import argparse
import json

from cli.bundle import TrainingBundle

from mobilenetv2ssd.core.fingerprint import Fingerprinter, Fingerprint
from mobilenetv2ssd.core.utils import initialize_run_metadata
from mobilenetv2ssd.core.config import load_config, PROJECT_ROOT
from mobilenetv2ssd.core.logger import build_logger_from_config, Logger
from mobilenetv2ssd.core.precision_config import PrecisionConfig
from mobilenetv2ssd.core.exceptions import GracefulShutdownException

from datasets.collate import create_training_dataset, create_validation_dataset
from datasets.transforms import build_train_transforms, build_validation_transforms
from datasets.base import create_dataset_from_config

from mobilenetv2ssd.models.ssd.orchestration.priors_orch import build_priors_from_config
from mobilenetv2ssd.models.factory import build_ssd_model

from training.optimizer import OptimizerFactory
from training.schedule import LearningRateSchedulerFactory
from training.amp import build_amp
from training.metrics import build_metrics_from_config
from training.ema import build_ema, EMA
from training.checkpoints import build_checkpoint_manager
from training.resume import collect_resumable_runs, select_run_interactive, validate_checkpoint_compatibility, discover_checkpoint
from training.shutdown import ShutdownHandler
from training.engine import fit

from infrastructure.s3_sync import build_s3_sync

import tensorflow as tf

FINGERPRINT_KEYS = [
    'input_size', 'num_classes', 'backbone', 'heads', 'priors',
    'loss', 'sampler', 'matcher', 'augmentation', 'optimizer',
    'scheduler', 'train', 'data', 'eval',
]

FINGERPRINT_EXCLUDES = {
    'train': {'diagnostics'},
    'eval': {'interval_epochs', 'visualization'},
    'data': {'loader'},
}

TRAINING_KEYS = ['backbone', 'heads', 'priors', 'loss', 'optimizer', 'scheduler', 
                 'train', 'data', 'augmentation', 'matcher', 'sampler', 'model']

def extract_training_keys(config: dict) -> dict:
    return {k: v for k, v in config.items() if k in TRAINING_KEYS}

def compute_fingerprint(config: dict[str, Any], git_commit: str | None = None):
    fingerprinting_dict = {}
    
    for key in FINGERPRINT_KEYS:
        if key not in config:
            continue
        
        # Adding the key to the fingerprint dict
        value = config[key]
        if key in FINGERPRINT_EXCLUDES and isinstance(config[key], dict):
            value = {k: v for k, v in config[key].items() if k not in FINGERPRINT_EXCLUDES[key]}
            
        fingerprinting_dict[key] = value
        
    if git_commit is not None:
        fingerprinting_dict['git_commit'] = git_commit
        
    return Fingerprinter().fingerprint(fingerprinting_dict)

def parse_args():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 SSD model.")
    parser.add_argument('--experiment_path', type=str, required=True, help='Path to the experiment configuration file.')
    parser.add_argument('--config_root', type=str, default=PROJECT_ROOT / "configs", help='Root directory for configuration files.')
    parser.add_argument('--git_commit', type=str, default=None, help='Git commit hash for fingerprinting.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint.')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training.')
    parser.add_argument('--print_config', action='store_true', help='Print the configuration and exit.')
    parser.add_argument('--dry_run', action='store_true', help='Perform a dry run without training.')
    parser.add_argument('--run_from', type=str, default=None, help='Resume directory from user')
    
    args = parser.parse_args()
        
    
    return {
        'experiment_path': Path(args.experiment_path),
        'config_root': Path(args.config_root),
        'git_commit': args.git_commit,
        'debug': args.debug,
        'resume': args.resume,
        'local_rank': args.local_rank,
        'print_config': args.print_config,
        'dry_run': args.dry_run,
        'resume_from': args.run_from
    }
    
def initialize_run_settings(args: dict[str, Any]):
    # TODO: Add stuff for distributed training, setting random seeds, etc.
    experiment_path = args['experiment_path']
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment configuration file not found at {experiment_path}")
    
    config_root = args['config_root']
    if not config_root.exists():
        raise FileNotFoundError(f"Configuration root directory not found at {config_root}")
    
    if args['debug']:
        print("Debug mode enabled. Setting up debug settings...")
        
    if args['print_config']:
        print("Print config mode enabled. Will print the configuration and exit.")
        # Print the configuration and exit
        config = load_config(experiment_path=experiment_path, config_root=config_root)
        print("Loaded Configuration:")
        print(config)
        exit(0)
        
    if args['dry_run']:
        print("Dry run mode enabled. Will perform a dry run without training.")
        
    # Now creating the logger and other settings
    
    config = load_config(experiment_path=experiment_path, config_root=config_root)
    
    if args['resume']:
        runs = collect_resumable_runs(Path(config['run']['root']))
        if runs:
            selected = select_run_interactive(runs)
            if selected is not None:
                with open(str(selected['experiment_dir'] / "config.json")) as config_file:
                    saved_config = json.load(config_file)
                    
                compatibility_flag, warnings = validate_checkpoint_compatibility(saved_config= saved_config, current_config= config)
                if not compatibility_flag:
                    print(f"Error in the restore path config and the current config : {warnings}")
                    exit(1)
                args['resume_checkpoint_path'] = selected['ckpt_path']
            else:
                print("No selected path found..............")
        else:
            print("No resumable runs found. Starting fresh.")
    elif args['resume_from']:
        run_path = Path(args['resume_from'])
        
        # Need to check if it is a directory and if it is a checkpoint path
        if run_path.is_dir():
            discovered_ckpt = discover_checkpoint(run_path)
            if discovered_ckpt is None:
                # Then the resume path is wrong its an error
                print(f"No checkpoint found in {run_path}")
                exit(1)
            
            args['resume_checkpoint_path'] = discovered_ckpt['ckpt_path']
        else:
            args['resume_checkpoint_path'] = run_path
    
    fingerprint = compute_fingerprint(config, git_commit=args['git_commit'])
    
    # Formatting the timestamp for logging and metadata
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Making the logger
    logger = build_logger_from_config(config=config, fingerprint= fingerprint)
    
    # Logging the initialization settings
    logger.info(f"Logger Initialized{'.'*20}")
    logger.info(f"Initialized run with configuration from {experiment_path} {'.'*20}")
    logger.info(f"Configuration root directory: {config_root} {'.'*20}")
    logger.info(f"Initialized run with fingerprint: {fingerprint.short} {'.'*20}")
    
    # Setting up the run metadata
    # Build the run metadata
    initialize_run_metadata(config= config, args= args, fingerprint=fingerprint, timestamp= timestamp)
    
    logger.info(f"Initialized run with fingerprint: {fingerprint.short} {'.'*20}")
    
    return config, logger, fingerprint, timestamp

def create_datasets(config: dict[str, Any], logger: Logger):
    
    # First creating the transform compose
    train_compose = build_train_transforms(config)
    validation_compose = build_validation_transforms(config)
    
    # Logging the transform creation step
    logger.info(f"Created transforms for training and validation datasets {'.'*20}")
    logger.info(f"Training transforms: {[transform.__class__.__name__ for transform in train_compose._transforms]} {'.'*20}")
    logger.info(f"Validation transforms: {[transform.__class__.__name__ for transform in validation_compose._transforms]} {'.'*20}")
    
    # Now creating the datasets
    # Training dataset has to always be created, but validation dataset is optional based on the config
    training_dataset = create_dataset_from_config(config= config, split= config['data']['train_split'])
    logger.info(f"Created {training_dataset.__class__.__name__} training dataset with {len(training_dataset)} samples {'.'*20}")
    logger.info(f"Train loop has {int(len(training_dataset) / config['data']['train']['batch_size'])} steps{'.'*20}")
    
    if config['eval']['eval_enabled']:
        validation_dataset = create_dataset_from_config(config= config, split= config['data']['val_split'])
        logger.info(f"Created {validation_dataset.__class__.__name__} validation dataset with {len(validation_dataset)} samples {'.'*20}")
        logger.info(f"Eval loop has {int(len(validation_dataset) / config['data']['val']['batch_size'])} steps{'.'*20}")
        
    # Leveraging the tf.data.Dataset API to create the training and validation datasets
    train_dataset = create_training_dataset(config= config, dataset= training_dataset, transform= train_compose)
    
    val_dataset = create_validation_dataset(config= config, dataset= validation_dataset, transform= validation_compose) if config['eval']['eval_enabled'] else None
    
    logger.info(f"Created training dataset with tf.data.Dataset API {'.'*20}")
    if val_dataset is not None:
        logger.info(f"Created validation dataset with tf.data.Dataset API {'.'*20}")
        
        
    return train_dataset, val_dataset
        
def create_priors(config: dict[str, Any], logger: Logger):
    # Creating the priors for the model once so it can be used over and over
    priors, priors_meta = build_priors_from_config(model_config = config)
    
    logger.info(f"Created Priors for SSD Model{'.'*20}")
    logger.info(f"Num of priors per layer: {priors_meta['number_of_anchors_per_layer'].numpy()}")
    logger.info(f"Total Num of priors: {priors_meta['total_number_of_anchors'].numpy()}")
    
    return priors, priors_meta

def create_optimizer(config: dict[str, Any], logger: Logger):
    # Creating the optimizers for the model
    
    learning_schedule = LearningRateSchedulerFactory.build(config= config)
    
    logger.info(f"Created Learning Schedule {learning_schedule.__class__.__name__}.....")
    
    optimizer = OptimizerFactory.build(config=config, learning_schedule= learning_schedule )
    
    logger.info(f"Created Optimizer {optimizer.__class__.__name__}.....")
    
    return optimizer
    
def create_amp(config: dict[str, Any], optimizer: tf.keras.optimizers.Optimizer, logger: Logger):
    
    amp = build_amp(config=config, optimizer= optimizer)
    
    precision_config = amp.make_precision_config()
    
    return amp, precision_config

def build_model(config: dict[str, Any], logger: Logger, priors_meta: dict[str, Any]):
    
    model = build_ssd_model(config= config, anchors_per_layer= priors_meta['anchors_per_cell'].numpy())
    logger.info(f"Model Name: {model.name}, Backbone: {model.backbone_type}, Input Shape: {model.input_shape}")
    
    return model

def create_ema(config: dict[str, Any], model: tf.keras.Model, logger: Logger):
    
    ema = build_ema(config= config, model= model)
    logger.info(f"EMA name: {ema.name}, decay: {ema._decay}, enabled: {ema._enabled}, warmup steps: {ema._warmup_steps}, update every: {ema._update_every}, eval_use_ema: {ema._eval_use_ema}")
    
    return ema

def create_build_checkpoint_manager(config: dict[str, Any], model: tf.keras.Model, logger: Logger, ema: EMA, optimizer: tf.keras.optimizers.Optimizer, fingerprint: Fingerprint | None = None):
    
    checkpoint_manager = build_checkpoint_manager(config= config, model= model, optimizer= optimizer, ema= ema, is_main_node = True, fingerprint= fingerprint, log_dir= logger.job_dir)
    logger.info(f"Created checkpoint manager at directory: {checkpoint_manager._checkpoint_directory}")
    
    logger.info(f"Building the optimizer...{'.'*20}")
    logger.info(f"Optimizer variables before build: {optimizer.variables}")
    checkpoint_manager.build_optimizer(var_group= model.trainable_variables)
    logger.success(f"Successfully built the optimizer...{'.'*20}")
    
    return checkpoint_manager

def initialize_framework(args: dict[str, Any]):
    
    config, logger, fingerprint, timestamp = initialize_run_settings(args)
    
    # Now creating the dataset
    logger.info(f"Creating datasets... {'.'*20}")
    train_dataset, val_dataset = create_datasets(config, logger)
    logger.info(f"Completed Dataset Creation...{'.'*20}")
    
    # Creating the priors
    logger.info(f"Creating priors...{'.'*20}")
    priors, priors_meta = create_priors(config= config, logger= logger)
    logger.info(f"Completed priors...{'.'*20}")
    
    # Creating an optimizer
    
    logger.info(f"Creating Optimizer...{'.'*20}")
    optimizer = create_optimizer(config, logger)
    logger.info(f"Completed Optimizer...{'.'*20}")
    
    # Creating an AMP
    logger.info(f"Creating AMP...{'.'*20}")
    amp, precision_config = create_amp(config, optimizer, logger)
    logger.info(f"Completed AMP...{'.'*20}")

    logger.success(f"Successfully Initialized Forced FP32 for {precision_config.get_forced_precision_fields()}...{'.'*20}")
    
    logger.info(f"Wrapping Optimizer for AMP...{'.'*20}")
    optimizer = amp.wrap_optimizer()
    logger.info(f"Optimizer class variable: {optimizer.__class__.__name__} with inner Optimizer: {optimizer.inner_optimizer.__class__.__name__ if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer) else 'NA'}")
    logger.success(f"Completed wrapping Optimizer for AMP...{'.'*20}")
    
    # Creating a metrics manager
    logger.info(f"Creating Metrics Manager...{'.'*20}")
    metrics_manager = build_metrics_from_config(config= config)
    logger.info(f"Metrics list {metrics_manager.metrics}")
    logger.success(f"Completed Metrics Manager Creation...{'.'*20}")
    
    # Building the model
    logger.info(f"Creating SSD Model...{'.'*20}")
    model = build_model(config= config, logger= logger, priors_meta= priors_meta)
    logger.success(f"Completed SSD Model Creation...{'.'*20}")
    
    # Building the EMA
    logger.info(f"Creating EMA...{'.'*20}")
    ema = create_ema(config= config, model= model, logger= logger)
    logger.success(f"Successfully created EMA...{'.'*20}")
    
    # Building Checkpoint Manager
    logger.info(f"Creating Checkpoint Manager...{'.'*20}")
    checkpoint_manager = create_build_checkpoint_manager(config= config, model= model, logger= logger, ema= ema, optimizer= optimizer, fingerprint= fingerprint)
    logger.success(f"Successfully Created Checkpoint Manager")
    
    # Building the S3 Sync Manager
    logger.info(f"Creating S3 Sync Client...{'.'*20}")
    s3_sync_client = build_s3_sync(config= config, logger= logger)
    logger.success(f"Successfully Created S3 Sync Client")
    
    if s3_sync_client is not None:
        # Getting the info for the upload
        experiment_name = config.get('experiment', {}).get('id', 'exp')
        experiment_subdir = f"{experiment_name}_{fingerprint.short}"

        # local_dir is absolute (for file operations)
        run_root = Path(config['run']['root'])
        experiment_directory = run_root / experiment_subdir

        # s3_sub_prefix is always relative, derived from actual directory name
        run_root_name = run_root.name  # e.g., "runs", "runs_log", "experiments"
        s3_sub_prefix = f"{run_root_name}/{experiment_subdir}"

        s3_sync_client.upload_directory(local_dir=experiment_directory, s3_sub_prefix=s3_sub_prefix)
    
    if args['dry_run']:
        logger.success(f"Dry Run Successfully Completed...{'.'*20}")
        exit(0)
        
    # Uploading the metadata to S3 for storage


    
    return TrainingBundle(logger= logger, fingerprint= fingerprint, run_dir= None, config= config, model= model, priors_cxcywh= priors, train_dataset= train_dataset, val_dataset= val_dataset, optimizer= optimizer, precision_config= precision_config, ema= ema, amp= amp, checkpoint_manager= checkpoint_manager, max_epochs= None, best_metric= None, metrics_manager= metrics_manager, s3_client= s3_sync_client)

def train(framework_opts: TrainingBundle, shutdown_handler: ShutdownHandler, resume_ckpt_path: Path | None = None):
    if resume_ckpt_path:
        # There is a path that the user passed and needs to restore from
        restore_state= framework_opts.checkpoint_manager.restore_from_directory(resume_ckpt_path)
    else:
        restore_state= framework_opts.checkpoint_manager.restore_latest()
    
    if not restore_state['restored']:
        start_epoch = restore_state['epoch']
        global_step = restore_state['global_step']
        best_metric = 0.0
    else:
        start_epoch = restore_state['epoch']
        global_step = restore_state['global_step']
        best_metric = restore_state['best_metric']
        
    framework_opts.start_epoch = start_epoch
    framework_opts.global_step = global_step
    framework_opts.best_metric = best_metric
    framework_opts.max_epochs = framework_opts.config['train']['epochs']
    
    framework_opts.logger.info(f"Starting Training Loop for {framework_opts.max_epochs}..{'.'*20}")
    
    # Now Fitting the model
    training_result = fit(config= framework_opts.config, 
                        model= framework_opts.model, 
                        priors_cxcywh= framework_opts.priors_cxcywh, 
                        train_dataset= framework_opts.train_dataset, 
                        validation_dataset= framework_opts.val_dataset, 
                        optimizer= framework_opts.optimizer, 
                        precision_config= framework_opts.precision_config, 
                        metrics_manager= framework_opts.metrics_manager,
                        checkpoint_manager= framework_opts.checkpoint_manager,
                        logger= framework_opts.logger,
                        ema= framework_opts.ema,
                        amp= framework_opts.amp,
                        start_epoch= start_epoch,
                        global_step= global_step,
                        max_epochs= framework_opts.config['train']['epochs'],
                        shutdown_handler= shutdown_handler,
                        s3_sync= framework_opts.s3_client)
        
    # Saving the Model weights:
    framework_opts.logger.save_model_weights(framework_opts.model,training_result, framework_opts.config, framework_opts.fingerprint, framework_opts.ema)
    
def execute_training():
    # Registering a handler
    handler = ShutdownHandler()
    handler.register()
    
    framework_opts = None
    args = None
    exit_code = 0
    
    try:
        args = parse_args()
    
        framework_opts = initialize_framework(args= args)
    
        framework_opts.logger.success(f"Completed the initialization stage for the framework...{'.'*20}")
        resume_path = args.get('resume_checkpoint_path', None)
        train(framework_opts= framework_opts, shutdown_handler= handler, resume_ckpt_path= resume_path)
    except GracefulShutdownException as err:
        exit_code= 128 + err.signal_number
    except Exception as err:
        exit_code= 1
        if framework_opts is not None:
            framework_opts.logger.error(f"Training failed with error: {err}")
        
        traceback.print_exc()
    finally:
        if framework_opts is not None:
            experiment_name = framework_opts.config.get('experiment', {}).get('id', 'exp')
            experiment_subdir = f"{experiment_name}_{framework_opts.fingerprint.short}"

            # local paths are absolute
            run_root = Path(framework_opts.config['run']['root'])
            experiment_directory = run_root / experiment_subdir
            status_path = experiment_directory / "status.json"

            with open(status_path, 'w') as file:
                json.dump({'status': "success" if exit_code == 0 else "failed"}, file)

            if framework_opts.s3_client is not None:
                # s3_sub_prefix is always relative, derived from actual directory name
                run_root_name = run_root.name
                s3_sub_prefix = f"{run_root_name}/{experiment_subdir}"
                framework_opts.s3_client.upload_directory(local_dir=experiment_directory, s3_sub_prefix=s3_sub_prefix)

            framework_opts.logger.close()
        
        handler.unregister()
        
    return exit_code
        

if __name__ == "__main__":
    
   sys.exit(execute_training())
    
    
         
    
    
    
    
    
    
    