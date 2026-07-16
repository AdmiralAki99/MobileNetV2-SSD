from onnxconverter_common import float16
import onnx
from pathlib import Path

def run_fp16_conversion(fp32_path: Path, output_path: Path):
    onnx.save(float16.convert_float_to_float16(onnx.load(fp32_path)), output_path)
    