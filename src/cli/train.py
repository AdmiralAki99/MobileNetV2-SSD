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
from mobilenetv2ssd.core.config import load_config, PROJECT_ROOT, _is_path_key
from mobilenetv2ssd.core.logger import build_logger_from_config, Logger
from mobilenetv2ssd.core.precision_config import PrecisionConfig
from mobilenetv2ssd.core.exceptions import GracefulShutdownException

from datasets.collate import create_training_dataset_from_tfrecords, create_validation_dataset_from_tfrecords, create_training_dataset, create_validation_dataset
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

from infrastructure.s3_sync import build_s3_sync, parse_bucket_uri, S3SyncClient
from infrastructure.util import download_checkpoint_from_s3
from infrastructure.dynamodb_ledger import build_dynamodb_ledger, ExperimentLedger, get_ec2_instance_id

import tensorflow as tf

FINGERPRINT_KEYS = [
    'input_size', 'num_classes', 'backbone', 'heads', 'priors',
    'loss', 'sampler', 'matcher', 'augmentation', 'optimizer',
    'scheduler', 'train', 'data', 'eval',
]

FINGERPRINT_EXCLUDES = {
    'train': {'diagnostics'},
    'eval': {'interval_epochs', 'visualization'},
    'data': {'loader', 'root', 'classes_file'}
}

TRAINING_KEYS = ['backbone', 'heads', 'priors', 'loss', 'optimizer', 'scheduler', 
                 'train', 'data', 'augmentation', 'matcher', 'sampler', 'model']

def _strip_path_keys(obj):
    if isinstance(obj, dict):
        return {k: _strip_path_keys(v) for k, v in obj.items() if not _is_path_key(k)}
    if isinstance(obj, list):
        return [_strip_path_keys(v) for v in obj]
    return obj

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
            
        fingerprinting_dict[key] = _strip_path_keys(value)
        
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
    parser.add_argument('--checkpoint_step', type=int, default=None, help='Specific checkpoint step to resume from. Defaults to the latest.')

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
        'resume_from': args.run_from,
        'checkpoint_step': args.checkpoint_step,
    }
    
def handle_experiment_ledger(experiment_ledger: ExperimentLedger, ledger_claimed: bool, experiment_id: str, fingerprint: Fingerprint, logger: Logger, s3_sync_client: S3SyncClient, args: dict[str, Any]):
    # Handle ledger
    if experiment_ledger is not None:
        state = experiment_ledger.get_experiment_state(experiment_id= experiment_id, fingerprint= fingerprint.short)
        if state is not None:
            status = state['status']
            
            match status:
                case 'success':
                    # The experiment has succeded
                    logger.info(f"Ledger: {experiment_id} already completed. Artifacts: {state.get('artifact_s3_path')}. Exiting.")
                    sys.exit(0)
                case 'running':
                    logger.info(f"Ledger: {experiment_id} already running on {state.get('ec2_instance')}. Exiting.")
                    sys.exit(0)
                case 'pending':
                    # Getting the EC2 instance
                    instance_id = get_ec2_instance_id()
                    
                    # Now checking if the ledger can claim
                    if not experiment_ledger.claim_experiment(experiment_id= experiment_id, fingerprint= fingerprint.short, timestamp= logger.timestamp, instance_id= instance_id):
                        logger.info(f"Ledger: lost race to claim {experiment_id}. Exiting.")
                        sys.exit(0)
                        
                    # It is claimed if it did not fail
                    ledger_claimed = True
                    logger.info(f"Ledger: claimed {experiment_id}/{fingerprint.short} (was {status})")
                case 'failed':
                    # The run failed so attempting to claim the experiment
                    instance_id = get_ec2_instance_id()
                    if not experiment_ledger.claim_experiment(experiment_id= experiment_id, fingerprint= fingerprint.short, timestamp= logger.timestamp, instance_id= instance_id):
                        logger.info(f"Ledger: lost race to claim {experiment_id}. Exiting.")
                        sys.exit(0)
                        
                    # It is claimed if it did not fail
                    ledger_claimed = True
                    logger.info(f"Ledger: claimed {experiment_id}/{fingerprint.short} (was {status})")
                        
                    s3_path = state.get('checkpoint_s3_path')
                    if s3_path and not args.get('resume_checkpoint_path') and s3_sync_client:
                        logger.info(f"Ledger: previous run failed with checkpoint at {s3_path}. Resuming.")
                        # Resuming if it exists
                        _, s3_prefix = parse_bucket_uri(s3_path)
                        local_dir, actual_step = download_checkpoint_from_s3(s3_client= s3_sync_client,s3_checkpoint_prefix= s3_prefix, checkpoint_step= None)
                        if local_dir:
                            # Discovering checkpoint
                            discovered_checkpoint = discover_checkpoint(local_dir, target_step= actual_step)
                            if discovered_checkpoint:
                                args['resume_checkpoint_path'] = discovered_checkpoint['ckpt_path']
                                logger.info(f"Ledger: will resume from step {discovered_checkpoint['step']}")    
        else:
            # Logger is None so there is not targeting
            logger.warning(f"Ledger: no entry for {experiment_id}/{fingerprint.short}. Proceeding without tracking.")
       
    return args, ledger_claimed
    
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
    
    # Compute Fingerprint
    fingerprint = compute_fingerprint(config, git_commit=args['git_commit'])
    
    # Making the logger
    logger = build_logger_from_config(config=config, fingerprint= fingerprint)
    
    # Logging the initialization settings
    logger.info(f"Logger Initialized{'.'*20}")
    logger.info(f"Initialized run with configuration from {experiment_path} {'.'*20}")
    logger.info(f"Configuration root directory: {config_root} {'.'*20}")
    logger.info(f"Initialized run with fingerprint: {fingerprint.short} {'.'*20}")
    
    # Building the S3 Sync Manager
    logger.info(f"Creating S3 Sync Client...{'.'*20}")
    s3_sync_client = build_s3_sync(config= config, logger= logger)
    logger.success(f"Successfully Created S3 Sync Client")
    
    # Creating a ledger
    experiment_id = config.get('experiment', {}).get('id','exp')
    experiment_ledger = build_dynamodb_ledger(config= config, logger= logger)
    ledger_claimed = False    
    
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
        resume_from = args['resume_from']

        # Check if it's an S3 path - download checkpoint files first
        if resume_from.startswith("s3://"):
            print(f"Downloading checkpoint from S3: {resume_from}")
            if s3_sync_client is None:
                print("S3 client not configured. Cannot download from S3.")
                exit(1)

            # Extract the S3 prefix (everything after bucket/)
            # e.g., s3://bucket/runs/exp001/logs/.../checkpoints/last -> runs/exp001/logs/.../checkpoints/last

            _, s3_prefix = parse_bucket_uri(resume_from)

            target_step = args.get('checkpoint_step')
            local_dir, actual_step = download_checkpoint_from_s3(s3_sync_client, s3_prefix, checkpoint_step=target_step)
            if local_dir is None:
                print(f"Failed to download checkpoint from {resume_from}")
                exit(1)
            discovered_ckpt = discover_checkpoint(local_dir, target_step=actual_step)
            if discovered_ckpt is None:
                step_hint = f" at step {target_step}" if target_step else ""
                print(f"No checkpoint found{step_hint} in downloaded files at {local_dir}")
                exit(1)

            args['resume_checkpoint_path'] = discovered_ckpt['ckpt_path']
            print(f"Resuming from step {discovered_ckpt['step']} (downloaded to {local_dir})")
        else:
            run_path = Path(resume_from)

            # Need to check if it is a directory and if it is a checkpoint path
            if run_path.is_dir():
                target_step = args.get('checkpoint_step')
                discovered_ckpt = discover_checkpoint(run_path, target_step=target_step)
                if discovered_ckpt is None:
                    step_hint = f" at step {target_step}" if target_step else ""
                    print(f"No checkpoint found{step_hint} in {run_path}")
                    exit(1)

                args['resume_checkpoint_path'] = discovered_ckpt['ckpt_path']
            else:
                args['resume_checkpoint_path'] = run_path
    
    # Formatting the timestamp for logging and metadata
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    args, ledger_claimed = handle_experiment_ledger(experiment_ledger= experiment_ledger, ledger_claimed= ledger_claimed, experiment_id= experiment_id, fingerprint= fingerprint, logger= logger, s3_sync_client= s3_sync_client, args= args)
    
    # Setting up the run metadata
    # Build the run metadata
    initialize_run_metadata(config= config, args= args, fingerprint=fingerprint, timestamp= timestamp)
    
    logger.info(f"Initialized run with fingerprint: {fingerprint.short} {'.'*20}")
    
    return config, logger, fingerprint, s3_sync_client, experiment_ledger, ledger_claimed

def create_datasets(config: dict[str, Any], logger: Logger):
    
    # First creating the transform compose
    train_compose = build_train_transforms(config)
    validation_compose = build_validation_transforms(config)
    
    # Logging the transform creation step
    logger.info(f"Created transforms for training and validation datasets {'.'*20}")
    logger.info(f"Training transforms: {[transform.__class__.__name__ for transform in train_compose._transforms]} {'.'*20}")
    logger.info(f"Validation transforms: {[transform.__class__.__name__ for transform in validation_compose._transforms]} {'.'*20}")
    
    use_tfrecords = config['data'].get('tfrecords',{}).get('enabled', False)
    
    # Splitting the data ingestion to be the raw data (Slow speed) or TFRecords Shards (GPU Optimized)
    if not use_tfrecords:
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
    else:
        train_shard_dir = Path(config['data']['root']) / "shards" / config['data']['train_split']
        train_shard_paths = [str(path) for path in train_shard_dir.iterdir() if path.is_file()]
        
        train_dataset = create_training_dataset_from_tfrecords(config= config, shard_paths= train_shard_paths, transform= train_compose)
        metadata_dataset = create_dataset_from_config(config= config, split= config['data']['train_split'])
        logger.info(f"Created {metadata_dataset.__class__.__name__} training dataset with {len(train_shard_paths)} shards {'.'*20}")
        logger.info(f"Created {metadata_dataset.__class__.__name__} training dataset with {len(metadata_dataset)} samples {'.'*20}")
        logger.info(f"Train loop has {int(len(metadata_dataset) // config['data']['train']['batch_size'])} steps{'.'*20}")
        
        if config['eval']['eval_enabled']:
            val_shard_dir = Path(config['data']['root']) / "shards" / config['data']['val_split']
            val_shard_paths = [str(path) for path in val_shard_dir.iterdir() if path.is_file()]
            val_dataset = create_validation_dataset_from_tfrecords(config= config, shard_paths= val_shard_paths, transform= validation_compose)
            
            val_metadata_dataset = create_dataset_from_config(config= config, split= config['data']['val_split'])
            logger.info(f"Created {val_metadata_dataset.__class__.__name__} validation dataset with {len(val_shard_paths)} shards {'.'*20}")
            logger.info(f"Created {val_metadata_dataset.__class__.__name__} validation dataset with {len(val_metadata_dataset)} samples {'.'*20}")
            logger.info(f"Eval loop has {int(len(val_metadata_dataset) // config['data']['val']['batch_size'])} steps{'.'*20}")
        else:
            val_dataset = None      
        
        logger.info(f"Created training dataset with tf.data.Dataset API {'.'*20}")
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
    
    config, logger, fingerprint, s3_sync_client, experiment_ledger, ledger_claimed = initialize_run_settings(args)
    
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
    
    return TrainingBundle(logger= logger, fingerprint= fingerprint, run_dir= None, config= config, model= model, priors_cxcywh= priors, train_dataset= train_dataset, val_dataset= val_dataset, optimizer= optimizer, precision_config= precision_config, ema= ema, amp= amp, checkpoint_manager= checkpoint_manager, max_epochs= None, best_metric= None, metrics_manager= metrics_manager, s3_client= s3_sync_client, experiment_ledger= experiment_ledger, ledger_claimed= ledger_claimed)

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
    
    framework_opts.logger.info(f"Step: {global_step}, Start_epoch: {start_epoch}, Best Metric: {best_metric}")
        
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
                        s3_sync= framework_opts.s3_client,
                        experiment_ledger= framework_opts.experiment_ledger,
                        fingerprint_short= framework_opts.fingerprint.short)
        
    # Saving the Model weights:
    framework_opts.logger.save_model_weights(framework_opts.model,training_result, framework_opts.config, framework_opts.fingerprint, framework_opts.ema)

    # Upload final artifacts (weights, summary) to S3 artifact bucket
    if framework_opts.s3_client is not None:
        run_root = Path(framework_opts.config['run']['root'])
        log_dir = framework_opts.logger.job_dir
        framework_opts.s3_client.upload_final_artifacts(log_dir, run_root)
        framework_opts.logger.success("Final artifacts uploaded to S3 artifact bucket")
        
    return training_result

def execute_training():
    # Registering a handler
    handler = ShutdownHandler()
    handler.register()
    
    framework_opts = None
    args = None
    exit_code = 0
    training_result = None
    
    try:
        args = parse_args()
    
        framework_opts = initialize_framework(args= args)
    
        framework_opts.logger.success(f"Completed the initialization stage for the framework...{'.'*20}")
        resume_path = args.get('resume_checkpoint_path', None)
        training_result = train(framework_opts= framework_opts, shutdown_handler= handler, resume_ckpt_path= resume_path)
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
                
            # Now handling the ledger
            if framework_opts.experiment_ledger is not None and framework_opts.ledger_claimed:
                # The experiment was claimed and there is a ledger
                experiment_id = framework_opts.config.get('experiment', {}).get('id','exp')
                fingerprint_short = framework_opts.fingerprint.short
                
                # Getting the state of the experiment once more
                current_state = framework_opts.experiment_ledger.get_experiment_state(experiment_id= experiment_id, fingerprint= fingerprint_short)
                checkpoint_s3_path = (current_state or {}).get('checkpoint_s3_path','')
                total_steps = (current_state or {}).get('total_steps', framework_opts.global_step)
                # Now checking the exit code
                if exit_code == 0 and training_result is not None:
                    # Marking the experiment a success
                    framework_opts.experiment_ledger.mark_success(experiment_id= experiment_id, fingerprint= fingerprint_short, checkpoint_s3_path= checkpoint_s3_path, artifact_s3_path= '', best_epoch= 0, total_steps= training_result.get('global_step',0), best_metric= float(training_result.get('best_metric', 0.0)))
                    
                    # Logging the experiment a sucess
                    framework_opts.logger.info(f"Ledger: marked {experiment_id} as success")
                else:
                    # The experiment failed so the reason needs to be given
                    reason = 'spot_preemption' if exit_code >= 128 else 'training_error'
                    framework_opts.experiment_ledger.mark_failure(experiment_id= experiment_id, fingerprint= fingerprint_short, checkpoint_s3_path= checkpoint_s3_path, total_steps= total_steps, reason= reason)

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
    
    
         
    
    
    
    
    
    
    