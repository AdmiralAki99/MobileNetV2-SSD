import json, tempfile, os
import yaml
from datetime import datetime, timezone

from mobilenetv2ssd.core.config import load_config
from .fingerprint import compute_fingerprint
from ..services.ledger import get_ledger
from ..config import EXPERIMENT_BUCKET, REGION
import boto3


def register_experiments(config: dict, task_type: str = "detector", git_commit: str | None = None):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as file:
        json.dump(config, file)
        temp_path = file.name
    try:
        resolved = load_config(temp_path)
    finally:
        os.unlink(temp_path)

    # Fingerprint
    fp = compute_fingerprint(resolved, git_commit=git_commit)
    exp_id = resolved.get("experiment", {}).get("id", "exp")

    # library and resolves locally in its own context
    config_key = f"experiments/{exp_id}_{fp.short}.yaml"
    boto3.client("s3", region_name=REGION).put_object(
        Bucket=EXPERIMENT_BUCKET,
        Key=config_key,
        Body=yaml.safe_dump(config).encode(),
    )
    
    # Building the ledger item
    item = {
        "experiment_id": exp_id,
        "fingerprint": fp.short,
        "fingerprint_hex": fp.hex,
        "status": "pending",
        "task_type": task_type,
        "config_ref": config_key,
        "priority": resolved.get("experiment",{}).get("priority",0),
        "instance_type": resolved.get("infrastructure", {}).get("instance_type", "NA"),
        "use_tfrecords": resolved.get("data", {}).get("tfrecords", {}).get("enabled", False),
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write to the DynamoDB ledger
    created = get_ledger().register_experiment(item=item)
    
    return {"experiment_id": exp_id, "fingerprint": fp.short, "config_ref": config_key, "created": created}

def preview_experiment(config: dict, git_commit: str | None = None):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        json.dump(config, f)
        temp_path = f.name
    try:
        resolved = load_config(temp_path)
    finally:
        os.unlink(temp_path)
    fp = compute_fingerprint(resolved, git_commit=git_commit)
    return {
        "thin_yaml": yaml.safe_dump(config, sort_keys=False),
        "resolved": resolved,
        "fingerprint": fp.short,
        "experiment_id": resolved.get("experiment", {}).get("id", "exp"),
    }   