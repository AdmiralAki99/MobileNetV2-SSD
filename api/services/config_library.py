import boto3
from pathlib import Path

from ..config import EXPERIMENT_BUCKET, REGION, CONFIGS_DIR

CONFIG_LIBRARY_PREFIX = "config-library/"
LOCAL_CONFIG_DIR = Path("/app/configs")


def upload_config(local_path: Path, relative_path: str) -> None:

    s3 = boto3.client("s3", region_name=REGION)
    key = CONFIG_LIBRARY_PREFIX + relative_path.replace("\\", "/")
    s3.upload_file(str(local_path), EXPERIMENT_BUCKET, key)


def sync_config_library():

    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=EXPERIMENT_BUCKET, Prefix=CONFIG_LIBRARY_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(CONFIG_LIBRARY_PREFIX):]
            if not rel:
                continue
            dest = LOCAL_CONFIG_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(EXPERIMENT_BUCKET, key, str(dest))
