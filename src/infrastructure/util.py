from pathlib import Path
from typing import Any
import tempfile
import re

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

def download_checkpoint_from_s3(s3_client: Any, s3_checkpoint_prefix: str, checkpoint_step: int | None = None):
    # s3_client: S3SyncClient instance
    # s3_checkpoint_prefix: S3 path to checkpoint directory
    # checkpoint_step: specific step to download; if None or not found, uses the latest available

    if s3_client is None:
        return None, None

    # List available keys to find which steps exist without downloading everything
    available_keys = s3_client.list_keys(s3_checkpoint_prefix)
    steps = []
    for key in available_keys:
        m = re.search(r'ckpt-(\d+)\.index$', Path(key).name)
        if m:
            steps.append(int(m.group(1)))

    if not steps:
        return None, None

    # Pick target step: use requested step if it exists, otherwise fall back to latest
    if checkpoint_step is not None and checkpoint_step in steps:
        target = checkpoint_step
    else:
        if checkpoint_step is not None:
            print(f"  Step {checkpoint_step} not found in S3 (available: {sorted(steps)}). Using latest.")
        target = max(steps)

    prefix = f"ckpt-{target}."

    def key_filter(relative_path: str) -> bool:
        name = Path(relative_path).name
        return name.startswith(prefix) or name == "checkpoint"

    local_dir = Path(tempfile.mkdtemp(prefix="s3_checkpoint_"))
    success = s3_client.download_directory(s3_sub_prefix=s3_checkpoint_prefix, local_dir=local_dir, key_filter=key_filter)

    if not success:
        return None, None

    return local_dir, target
