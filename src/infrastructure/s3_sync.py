import boto3
from pathlib import Path
from typing import Any

class S3SyncClient:
    def __init__(self, bucket: str, base_prefix: str, logger: Any = None):
        self._client = boto3.client('s3')
        self._bucket = bucket
        self._base_prefix = base_prefix
        self._logger = logger
    
    def upload_file(self, local_file: Path, s3_key: str):
        """Upload a single file to S3"""
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
        pass
    
    
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