from pathlib import Path
import traceback

from deploy import load_deploy_config
from mobilenetv2ssd.core.config import PROJECT_ROOT
from infrastructure.dynamodb_ledger import ExperimentLedger
from infrastructure.s3_sync import S3SyncClient
from .export import run_export


def run_savedmodel_stage(experiment_id, fingerprint, deploy_config: Path, ledger: ExperimentLedger, output_directory: Path | None, s3_client: S3SyncClient):
    try:
        record = ledger.get_experiment_state(experiment_id=experiment_id, fingerprint=fingerprint)
        if not record:
            raise RuntimeError(f"No ledger record found for {experiment_id}/{fingerprint}")
        
        config = load_deploy_config(deploy_config)
        
        checkpoint_s3_path = record["checkpoint_s3_path"]
        if run_export(deploy_config=deploy_config, checkpoint_path=checkpoint_s3_path, output_dir=output_directory) != 0:
            raise RuntimeError("SavedModel export failed")
        
        # Resolcing where the output path
        if output_directory:
            saved_model_dir = output_directory / "saved_model"
        else:
            saved_model_dir = PROJECT_ROOT / config["deploy"]["saved_model_path"]
            
        artifact_prefix = record["artifact_s3_path"]
        s3_client.upload_to_artifact_bucket(saved_model_dir, f"{artifact_prefix}/saved_model")
        
        return f"{artifact_prefix}/saved_model"
    
    except Exception:
        traceback.print_exc()
        return None
        
        