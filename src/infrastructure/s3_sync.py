import boto3
import sys
from pathlib import Path
from typing import Any


class ProgressBar:

    def __init__(self, filename: str, total_bytes: int):
        self._filename = filename
        self._total = total_bytes
        self._downloaded = 0

    def __call__(self, bytes_transferred):
        self._downloaded += bytes_transferred
        pct = (self._downloaded / self._total * 100) if self._total > 0 else 0
        size_mb = self._total / (1024 * 1024)
        done_mb = self._downloaded / (1024 * 1024)
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r  {self._filename}: {bar} {done_mb:.1f}/{size_mb:.1f} MB ({pct:.0f}%)")
        sys.stdout.flush()
        if self._downloaded >= self._total:
            sys.stdout.write("\n")
            sys.stdout.flush()

class S3SyncClient:
    def __init__(self, checkpoint_bucket: str, artifact_bucket: str = None, dataset_bucket: str = None, logger: Any = None):
        # checkpoint_bucket: S3 URI for checkpoints (required)
        # artifact_bucket: S3 URI for final artifacts (weights, summaries)
        # dataset_bucket: S3 URI for datasets
        # logger: Logger instance
        
        self._client = boto3.client('s3')
        self._logger = logger

        # Parse checkpoint bucket (required)
        self._checkpoint_bucket, self._checkpoint_prefix = parse_bucket_uri(checkpoint_bucket)

        # Parse artifact bucket (optional)
        if artifact_bucket:
            self._artifact_bucket, self._artifact_prefix = parse_bucket_uri(artifact_bucket)
        else:
            self._artifact_bucket, self._artifact_prefix = None, None

        # Parse dataset bucket (optional)
        if dataset_bucket:
            self._dataset_bucket, self._dataset_prefix = parse_bucket_uri(dataset_bucket)
        else:
            self._dataset_bucket, self._dataset_prefix = None, None

        # Legacy support - default to checkpoint bucket
        self._bucket = self._checkpoint_bucket
        self._base_prefix = self._checkpoint_prefix
    
    def upload_file(self, local_file: Path, s3_key: str):
        try:
            local_file = Path(local_file)
            if not local_file.is_file():
                return

            # Combine base_prefix with s3_key
            full_key = f"{self._base_prefix}/{s3_key}".strip("/") if self._base_prefix else s3_key

            self._client.upload_file(
                Filename=str(local_file),
                Bucket=self._bucket,
                Key=full_key
            )

            if self._logger:
                self._logger.info(f"Uploaded {full_key}")
            else:
                print(f"✓ Uploaded: {full_key}")

        except Exception as err:
            if self._logger:
                self._logger.warning(f"S3 upload file failed: {err}")

    def upload_directory(self, local_dir: Path, s3_sub_prefix: str):
        # Trying the upload file

        try:

            local_dir = Path(local_dir)

            for file_path in local_dir.glob("*"):
                if file_path.is_dir():
                    continue

                relative_path = file_path.relative_to(local_dir)

                # Getting the parts
                parts = [self._base_prefix, s3_sub_prefix, str(relative_path)]

                # Now breaking them down
                s3_key= "/".join(part for part in parts if part)

                # Actually upload to S3
                self._client.upload_file(
                    Filename=str(file_path),
                    Bucket=self._bucket,
                    Key=s3_key
                )

                if self._logger:
                    self._logger.info(f"Uploaded {s3_key}")
                else:
                    print(f"✓ Uploaded: {s3_key}")

        except Exception as err:
            if self._logger:
                self._logger.warning(f"S3 upload failed: {err}. Checkpoint saved locally only.")
    
    def list_keys(self, s3_sub_prefix: str) -> list:
        full_prefix = f"{self._base_prefix}/{s3_sub_prefix}".strip("/") if self._base_prefix else s3_sub_prefix
        paginator = self._client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self._bucket, Prefix=full_prefix)
        keys = []
        for page in pages:
            for obj in page.get('Contents', []):
                relative_path = obj['Key'][len(full_prefix):].lstrip("/")
                if relative_path:
                    keys.append(relative_path)
        return keys

    def download_directory(self, s3_sub_prefix: str, local_dir: Path, key_filter=None):

        try:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)

            # Build the full prefix to list objects
            full_prefix = f"{self._base_prefix}/{s3_sub_prefix}".strip("/") if self._base_prefix else s3_sub_prefix

            # List all objects under the prefix
            paginator = self._client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self._bucket, Prefix=full_prefix)

            downloaded = 0
            for page in pages:
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']

                    # Get the relative path from the prefix
                    relative_path = s3_key[len(full_prefix):].lstrip("/")
                    if not relative_path:
                        continue

                    if key_filter and not key_filter(relative_path):
                        continue

                    local_file = local_dir / relative_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    file_size = obj.get('Size', 0)
                    progress = ProgressBar(relative_path, file_size)

                    self._client.download_file(
                        Bucket=self._bucket,
                        Key=s3_key,
                        Filename=str(local_file),
                        Callback=progress
                    )
                    downloaded += 1

            if self._logger:
                self._logger.info(f"Downloaded {downloaded} files from s3://{self._bucket}/{full_prefix}")

            return downloaded > 0

        except Exception as err:
            if self._logger:
                self._logger.warning(f"S3 download failed: {err}")
            else:
                print(f"S3 download failed: {err}")
            return False

    def upload_training_artifacts(self, log_directory: Path, run_root: Path):
        # log_directory: Path to timestamped log directory (e.g., logs/20260214_123456/)
        # run_root: Path to run root directory (e.g., runs/)
        
        if self._checkpoint_bucket is None:
            return

        # Upload checkpoint files from last/
        last_checkpoint_dir = log_directory / "checkpoints" / "last"
        if last_checkpoint_dir.exists():
            for ckpt_file in last_checkpoint_dir.glob("ckpt-*"):
                if ckpt_file.is_file():
                    relative_path = ckpt_file.relative_to(run_root.parent)
                    s3_key = f"{self._checkpoint_prefix}/{relative_path}".strip("/").replace("\\", "/")
                    self._client.upload_file(
                        Filename=str(ckpt_file),
                        Bucket=self._checkpoint_bucket,
                        Key=s3_key
                    )
                    if self._logger:
                        self._logger.info(f"Uploaded {s3_key}")

        # Upload checkpoint files from best/
        best_checkpoint_dir = log_directory / "checkpoints" / "best"
        if best_checkpoint_dir.exists():
            for ckpt_file in best_checkpoint_dir.glob("ckpt-*"):
                if ckpt_file.is_file():
                    relative_path = ckpt_file.relative_to(run_root.parent)
                    s3_key = f"{self._checkpoint_prefix}/{relative_path}".strip("/").replace("\\", "/")
                    self._client.upload_file(
                        Filename=str(ckpt_file),
                        Bucket=self._checkpoint_bucket,
                        Key=s3_key
                    )
                    if self._logger:
                        self._logger.info(f"Uploaded {s3_key}")

        # Upload TensorBoard events (in log directory)
        for tb_file in log_directory.glob("events.out.tfevents.*"):
            if tb_file.is_file():
                relative_path = tb_file.relative_to(run_root.parent)
                s3_key = f"{self._checkpoint_prefix}/{relative_path}".strip("/").replace("\\", "/")
                self._client.upload_file(
                    Filename=str(tb_file),
                    Bucket=self._checkpoint_bucket,
                    Key=s3_key
                )

        # Upload TensorBoard events (in tensorboard subdirectory if exists)
        tb_subdir = log_directory / "tensorboard"
        if tb_subdir.exists():
            for tb_file in tb_subdir.glob("events.out.tfevents.*"):
                if tb_file.is_file():
                    relative_path = tb_file.relative_to(run_root.parent)
                    s3_key = f"{self._checkpoint_prefix}/{relative_path}".strip("/").replace("\\", "/")
                    self._client.upload_file(
                        Filename=str(tb_file),
                        Bucket=self._checkpoint_bucket,
                        Key=s3_key
                    )

        # Upload training.log
        training_log = log_directory / "training.log"
        if training_log.exists():
            relative_path = training_log.relative_to(run_root.parent)
            s3_key = f"{self._checkpoint_prefix}/{relative_path}".strip("/").replace("\\", "/")
            self._client.upload_file(
                Filename=str(training_log),
                Bucket=self._checkpoint_bucket,
                Key=s3_key
            )

        # Upload metric history
        metric_file = log_directory / "metric_history.json"
        if metric_file.exists():
            relative_path = metric_file.relative_to(run_root.parent)
            s3_key = f"{self._checkpoint_prefix}/{relative_path}".strip("/").replace("\\", "/")
            self._client.upload_file(
                Filename=str(metric_file),
                Bucket=self._checkpoint_bucket,
                Key=s3_key
            )

        # Upload checkpoint metadata file
        checkpoint_metadata = log_directory / "checkpoints" / "checkpoint"
        if checkpoint_metadata.exists():
            relative_path = checkpoint_metadata.relative_to(run_root.parent)
            s3_key = f"{self._checkpoint_prefix}/{relative_path}".strip("/").replace("\\", "/")
            self._client.upload_file(
                Filename=str(checkpoint_metadata),
                Bucket=self._checkpoint_bucket,
                Key=s3_key
            )

    def upload_final_artifacts(self, log_directory: Path, run_root: Path):
        #  log_directory: Path to timestamped log directory (e.g., logs/20260214_123456/)
        #  run_root: Path to run root directory (e.g., runs/)
        
        if self._artifact_bucket is None:
            if self._logger:
                self._logger.warning("Artifact bucket not configured. Skipping final artifact upload.")
            return

        # Upload weights directory
        weights_dir = log_directory / "weights"
        if weights_dir.exists():
            for weight_file in weights_dir.glob("*"):
                if weight_file.is_file():
                    relative_path = weight_file.relative_to(run_root.parent)
                    s3_key = f"{self._artifact_prefix}/{relative_path}".strip("/").replace("\\", "/")
                    self._client.upload_file(
                        Filename=str(weight_file),
                        Bucket=self._artifact_bucket,
                        Key=s3_key
                    )
                    if self._logger:
                        self._logger.info(f"Uploaded final artifact: {s3_key}")

        # Upload training summary
        summary_file = log_directory.parent / "training_summary.json"
        if summary_file.exists():
            relative_path = summary_file.relative_to(run_root.parent)
            s3_key = f"{self._artifact_prefix}/{relative_path}".strip("/").replace("\\", "/")
            self._client.upload_file(
                Filename=str(summary_file),
                Bucket=self._artifact_bucket,
                Key=s3_key
            )
            if self._logger:
                self._logger.info(f"Uploaded training summary: {s3_key}")


def parse_bucket_uri(uri: str):
    # Separate the s3://
    path = uri.removeprefix("s3://")
    
    if "/" in path:
        # Split the parts for the path
        bucket, prefix = path.split("/",1)
    else:
        bucket, prefix = path, ""
        
    return bucket, prefix


def build_s3_sync(config: dict, logger: Any = None):

    storage_config = config.get('infrastructure', {}).get('storage', {})

    checkpoint_bucket = storage_config.get('checkpoint_bucket')
    artifact_bucket = storage_config.get('artifact_bucket')
    dataset_bucket = storage_config.get('dataset_bucket')

    # Checkpoint bucket is required
    if not checkpoint_bucket:
        if logger:
            logger.warning("No checkpoint bucket configured. S3 sync disabled.")
        return None

    try:
        client = S3SyncClient(
            checkpoint_bucket=checkpoint_bucket,
            artifact_bucket=artifact_bucket,
            dataset_bucket=dataset_bucket,
            logger=logger
        )
        return client
    except Exception as e:
        if logger:
            logger.warning(f"S3 sync unavailable: {e}. Running without remote backup.")

        return None