from pathlib import Path
import traceback

from deploy import load_deploy_config
from mobilenetv2ssd.core.config import PROJECT_ROOT
from infrastructure.s3_sync import S3SyncClient
from .export import run_export


def run_savedmodel_stage(
    checkpoint_s3_path: str,
    artifact_s3_path: str,
    deploy_config: Path,
    output_directory: Path | None,
    s3_client: S3SyncClient,
):
    try:
        config = load_deploy_config(deploy_config)

        if (
            run_export(deploy_config=deploy_config, checkpoint_path=checkpoint_s3_path, output_dir=output_directory)
            != 0
        ):
            raise RuntimeError("SavedModel export failed")

        if output_directory:
            saved_model_dir = output_directory / "saved_model"
        else:
            saved_model_dir = PROJECT_ROOT / config["deploy"]["saved_model_path"]

        s3_client.upload_to_artifact_bucket(saved_model_dir, f"{artifact_s3_path}/saved_model")

        return f"{artifact_s3_path}/saved_model"

    except Exception:
        traceback.print_exc()
        return None
