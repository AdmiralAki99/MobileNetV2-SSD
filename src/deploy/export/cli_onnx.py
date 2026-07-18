import argparse
import sys
from pathlib import Path

from .runner_onnx import run_onnx_bundle


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SavedModel to ONNX bundle (fp32/fp16/int8).")
    parser.add_argument("--deploy_config", type=str, required=True, help="Path to deploy config YAML.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for ONNX files. Defaults to paths in deploy config.",
    )
    parser.add_argument("--calibration_dir", type=str, required=True, help="Directory of images for INT8 calibration.")
    parser.add_argument("--num_calibration", type=int, default=100, help="Number of calibration images to use.")

    args = parser.parse_args()
    return {
        "deploy_config": Path(args.deploy_config),
        "output_dir": Path(args.output_dir) if args.output_dir else None,
        "calibration_dir": Path(args.calibration_dir),
        "num_calibration": args.num_calibration,
    }


def execute_onnx_bundle(args):
    paths = run_onnx_bundle(
        deploy_config_path=args["deploy_config"],
        output_directory=args["output_dir"],
        calibration_images_dir=args["calibration_dir"],
        num_calibration=args["num_calibration"],
    )

    if paths is None:
        print("ONNX bundle failed.")
        return 1

    print("ONNX bundle complete:")
    for precision, path in paths.items():
        print(f"  {precision}: {path}")

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(execute_onnx_bundle(args))
