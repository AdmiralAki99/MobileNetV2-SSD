from pathlib import Path
import traceback
from typing import Any

import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from PIL import Image
import numpy as np

from deploy import load_deploy_config
from mobilenetv2ssd.core.config import PROJECT_ROOT


def validate_onnx(onnx_path: Path, deploy_config: dict[str, Any]):
    # Creating an ingerence cession on the model
    onnx_session = ort.InferenceSession(str(onnx_path))

    # Printing out the inputs and outputs
    print("Inputs:", [(input_.name, input_.shape) for input_ in onnx_session.get_inputs()])
    print("Outputs:", [(output_.name, output_.shape) for output_ in onnx_session.get_outputs()])

    # Getting info from the config
    H, W = deploy_config["deploy"]["input"]["size"][0], deploy_config["deploy"]["input"]["size"][1]
    num_classes = deploy_config["deploy"]["classes"]["num_classes"]
    B = deploy_config["deploy"]["runtime"]["batch_size"]

    dummy_image = np.zeros([B, H, W, 3], dtype=np.float32)

    input_name = onnx_session.get_inputs()[0].name

    # Runnning the model
    raw_outputs = onnx_session.run(None, {input_name: dummy_image})

    outputs = onnx_session.get_outputs()

    results = {o.name: raw_outputs[i] for i, o in enumerate(outputs)}

    boxes = results["boxes"]
    scores = results["scores"]
    number_of_anchors = scores.shape[1]

    assert boxes.shape == (B, number_of_anchors, 4), f"Bad boxes shape: {boxes.shape}"
    assert scores.shape == (B, number_of_anchors, num_classes), f"Bad scores shape: {scores.shape}"


class _ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir: Path, input_name: str, H: int, W: int, number_of_images: int):
        extensions = ("*.jpg", "*.jpeg", "*.png")

        # Getting the supported extensions only
        paths = sorted(img for extension in extensions for img in calibration_dir.glob(extension))

        if not paths:
            raise FileNotFoundError(f"No images found in {calibration_dir}")

        # There is no error and there are images present

        self._paths = paths[:number_of_images]  # Need to slice the images if the directory has way too many
        self._input_name = input_name
        self._H = H
        self._W = W
        self._index = 0  # counter for the images
        print(f"Calibrating on {len(self._paths)} images from {calibration_dir}")

    def get_next(self):
        # This is the main method for supplying the images using a inbuilt generator

        if self._index >= len(self._paths):
            # The index should not go more than the length
            return None

        # Get the Image and preprocess it (Mirrors my Dataset loader)
        image = Image.open(self._paths[self._index]).convert("RGB")

        # Resize it into the format needed
        image = image.resize((self._W, self._H), Image.BILINEAR)

        # Normalize the image
        image = np.array(image, dtype=np.float32) / 255.0

        # Make it into a batch
        image = np.expand_dims(image, axis=0)

        if self._index % 50 == 0:
            print(f"  [{self._index + 1}/{len(self._paths)}]")

        self._index = self._index + 1

        return {self._input_name: image}  # ONNX requrires the keys to be similar to reference the info


def run_quantize(deploy_config: Path, calibration_images_dir: Path, num_calibration: int, output_path: Path | None):
    try:

        # Load the deploy config
        deploy_config = load_deploy_config(deploy_config)

        # Resolving the paths
        if output_path:
            onnx_path = output_path / "model.onnx"
            quantized_model_path = output_path / "model_int8.onnx"
        else:
            onnx_path = PROJECT_ROOT / deploy_config["deploy"]["onnx_path"]
            quantized_model_path = PROJECT_ROOT / deploy_config["deploy"]["quantized_onnx_path"]

        # Size for the input
        H, W = deploy_config["deploy"]["input"]["size"][0], deploy_config["deploy"]["input"]["size"][1]

        # Getting the number of images
        number_of_images = int(num_calibration)

        # Starting the session for quantization
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        input_name = session.get_inputs()[0].name

        del session

        print(f"Float32 model : {onnx_path}")
        print(f"Output        : {quantized_model_path}")
        print(f"Input         : {input_name}  ({H}x{W}x3 float32)")
        print("Quant format  : QDQ / INT8  (TensorRT-compatible)")

        # Creating that Calibration Reader
        calibration_reader = _ImageCalibrationReader(
            calibration_dir=calibration_images_dir,
            input_name=input_name,
            H=H,
            W=W,
            number_of_images=number_of_images,
        )

        # Quantizing the model
        quantize_static(
            model_input=str(onnx_path),
            model_output=str(quantized_model_path),
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )

        # Running a sanity check

        validate_onnx(onnx_path=quantized_model_path, deploy_config=deploy_config)

        print(f"PASS — quantized model saved to {quantized_model_path}")

        return 0

    except Exception:
        traceback.print_exc()
        return 1
