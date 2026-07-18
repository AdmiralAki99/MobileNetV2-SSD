import traceback
import subprocess
from pathlib import Path
import sys

from deploy import load_deploy_config
from mobilenetv2ssd.core.config import PROJECT_ROOT


def run_convert(deploy_config: Path, output_dir: Path | None):
    try:

        # Config Parameters
        deploy_config = load_deploy_config(deploy_config)
        if output_dir:
            model_save_path = output_dir / "saved_model"
            onnx_path = output_dir / "model.onnx"
        else:
            model_save_path = PROJECT_ROOT / deploy_config["deploy"]["saved_model_path"]
            onnx_path = PROJECT_ROOT / deploy_config["deploy"]["onnx_path"]

        opset = deploy_config["deploy"]["runtime"]["opset"]

        # Running the conversion process
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                str(model_save_path),
                "--output",
                str(onnx_path),
                "--opset",
                str(opset),
            ],
            check=True,
        )

        return 0

    except Exception:
        traceback.print_exc()
        return 1
