from dataclasses import dataclass
from typing import Any
from mobilenetv2ssd.core.logger import Logger
from mobilenetv2ssd.core.fingerprint import Fingerprint
from mobilenetv2ssd.core.precision_config import PrecisionConfig

from training.metrics import MetricsCollection
from training.checkpoints import CheckpointManager
from training.ema import EMA
from training.amp import AMPContext

from infrastructure.s3_sync import S3SyncClient
import tensorflow as tf


@dataclass
class TrainingBundle:
    
    # Determinism Factors 
    logger: Logger
    fingerprint: Fingerprint
    run_dir: str | None
    config: dict[str, Any]
    
    # Training Factors
    model: Any
    priors_cxcywh: Any
    train_dataset: Any
    val_dataset: Any
    optimizer: Any
    precision_config: PrecisionConfig
    ema: EMA
    amp: AMPContext
    
    # Metrics Factors
    metrics_manager: MetricsCollection
    
    # State Factors
    checkpoint_manager: CheckpointManager
    start_epoch: int = 0
    max_epochs: int | None = None
    global_step: int = 0
    best_metric: int | None = None
    
    # S3 Sync Client for storage
    s3_client: S3SyncClient | None = None
    
# @dataclass
# class InferenceBundle:

# @dataclass
# class DeploymentBundle: