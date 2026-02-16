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
    def __init__(self, bucket: str, base_prefix: str, logger: Any = None):
        self._client = boto3.client('s3')
        self._bucket = bucket
        self._base_prefix = base_prefix
        self._logger = logger
    
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
    
    def download_directory(self, s3_sub_prefix: str, local_dir: Path):

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
    bucket_uri = config.get('infrastructure',{}).get('storage',{}).get('checkpoint_bucket')
    
    # Check if the bucket exists
    if not bucket_uri:
        return None
    
    try:
    
        bucket, prefix = parse_bucket_uri(uri= bucket_uri)
        client = S3SyncClient(bucket= bucket, base_prefix= prefix, logger= logger)
        return client
    except Exception as e:
        if logger:
            logger.warning(f"S3 sync unavailable: {e}. Running without remote backup.")
            
        return None