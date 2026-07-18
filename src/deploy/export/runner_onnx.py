import traceback
from pathlib import Path

from deploy import load_deploy_config
from mobilenetv2ssd.core.config import PROJECT_ROOT
from .convert import run_convert
from .to_fp16 import run_fp16_conversion
from .quantize import run_quantize
from .validate import run_validate


def run_onnx_bundle(
    deploy_config_path: Path, output_directory: Path | None, calibration_images_dir: Path, num_calibration: int = 100
):
    try:
        config = load_deploy_config(deploy_config_path)

        # Resolve paths the same way each core does
        if output_directory:
            fp32_path = output_directory / "model.onnx"
            fp16_path = output_directory / "model_fp16.onnx"
            int8_path = output_directory / "model_int8.onnx"
        else:
            fp32_path = PROJECT_ROOT / config["deploy"]["onnx_path"]
            fp16_path = fp32_path.parent / "model_fp16.onnx"
            int8_path = PROJECT_ROOT / config["deploy"]["quantized_onnx_path"]

        # FP32
        if run_convert(deploy_config=deploy_config_path, output_dir=output_directory) != 0:
            raise RuntimeError("FP32 conversion failed")

        if run_validate(deploy_config=deploy_config_path, output_dir=output_directory) != 0:
            raise RuntimeError("FP32 validation failed")

        # FP16
        run_fp16_conversion(fp32_path=fp32_path, output_path=fp16_path)

        # INT8
        if (
            run_quantize(
                deploy_config=deploy_config_path,
                calibration_images_dir=calibration_images_dir,
                num_calibration=num_calibration,
                output_path=output_directory,
            )
            != 0
        ):
            raise RuntimeError("INT8 quantization failed")

        return {"fp32": fp32_path, "fp16": fp16_path, "int8": int8_path}

    except Exception:
        traceback.print_exc()
        return None
