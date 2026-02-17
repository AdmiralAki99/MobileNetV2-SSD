from pathlib import Path
from typing import Any
import tempfile

def upload_training_artifacts(s3_client: Any, log_directory: Path, run_root: Path):
    
    # s3_client: S3SyncClient instance
    # log_directory: Path to timestamped log directory (e.g., logs/20260214_123456/)
    # run_root: Path to run root directory (e.g., runs/)
    if s3_client is None:
        return

    # Upload checkpoint files from last/
    last_checkpoint_dir = log_directory / "checkpoints" / "last"
    if last_checkpoint_dir.exists():
        for ckpt_file in last_checkpoint_dir.glob("ckpt-*"):
            if ckpt_file.is_file():
                relative_path = ckpt_file.relative_to(run_root.parent)
                s3_key = str(relative_path).replace("\\", "/")
                s3_client.upload_file(local_file=ckpt_file, s3_key=s3_key)

    # Upload checkpoint files from best/
    best_checkpoint_dir = log_directory / "checkpoints" / "best"
    if best_checkpoint_dir.exists():
        for ckpt_file in best_checkpoint_dir.glob("ckpt-*"):
            if ckpt_file.is_file():
                relative_path = ckpt_file.relative_to(run_root.parent)
                s3_key = str(relative_path).replace("\\", "/")
                s3_client.upload_file(local_file=ckpt_file, s3_key=s3_key)

    # Upload TensorBoard events (in log directory)
    for tb_file in log_directory.glob("events.out.tfevents.*"):
        if tb_file.is_file():
            relative_path = tb_file.relative_to(run_root.parent)
            s3_key = str(relative_path).replace("\\", "/")
            s3_client.upload_file(local_file=tb_file, s3_key=s3_key)

    # Upload TensorBoard events (in tensorboard subdirectory if exists)
    tb_subdir = log_directory / "tensorboard"
    if tb_subdir.exists():
        for tb_file in tb_subdir.glob("events.out.tfevents.*"):
            if tb_file.is_file():
                relative_path = tb_file.relative_to(run_root.parent)
                s3_key = str(relative_path).replace("\\", "/")
                s3_client.upload_file(local_file=tb_file, s3_key=s3_key)

    # Upload training.log
    training_log = log_directory / "training.log"
    if training_log.exists():
        relative_path = training_log.relative_to(run_root.parent)
        s3_key = str(relative_path).replace("\\", "/")
        s3_client.upload_file(local_file=training_log, s3_key=s3_key)

    # Upload metric history
    metric_file = log_directory / "metric_history.json"
    if metric_file.exists():
        relative_path = metric_file.relative_to(run_root.parent)
        s3_key = str(relative_path).replace("\\", "/")
        s3_client.upload_file(local_file=metric_file, s3_key=s3_key)

    # Upload checkpoint metadata file
    checkpoint_metadata = log_directory / "checkpoints" / "checkpoint"
    if checkpoint_metadata.exists():
        relative_path = checkpoint_metadata.relative_to(run_root.parent)
        s3_key = str(relative_path).replace("\\", "/")
        s3_client.upload_file(local_file=checkpoint_metadata, s3_key=s3_key)

def upload_final_artifacts(s3_client: Any, log_directory: Path, run_root: Path):
    # s3_client: S3SyncClient instance configured for artifact bucket
    # log_directory: Path to timestamped log directory (e.g., logs/20260214_123456/)
    # run_root: Path to run root directory (e.g., runs/)
    if s3_client is None:
        return

    # Upload weights directory
    weights_dir = log_directory / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*"):
            if weight_file.is_file():
                relative_path = weight_file.relative_to(run_root.parent)
                s3_key = str(relative_path).replace("\\", "/")
                s3_client.upload_file(local_file=weight_file, s3_key=s3_key)

    # Upload training summary
    summary_file = log_directory.parent / "training_summary.json"
    if summary_file.exists():
        relative_path = summary_file.relative_to(run_root.parent)
        s3_key = str(relative_path).replace("\\", "/")
        s3_client.upload_file(local_file=summary_file, s3_key=s3_key)

def download_checkpoint_from_s3(s3_client: Any, s3_checkpoint_prefix: str):
    # s3_client: S3SyncClient instance
    # s3_checkpoint_prefix: S3 path to checkpoint directory
    # e.g., "runs/exp001_abc123/logs/20260214_123456/checkpoints/last"

    if s3_client is None:
        return None

    local_dir = Path(tempfile.mkdtemp(prefix="s3_checkpoint_"))
    success = s3_client.download_directory(s3_sub_prefix=s3_checkpoint_prefix, local_dir=local_dir)

    if not success:
        return None

    return local_dir
