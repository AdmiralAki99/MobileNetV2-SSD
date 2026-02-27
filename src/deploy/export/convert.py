import argparse
import traceback
import subprocess
from pathlib import Path
import sys

from deploy import load_deploy_config
from mobilenetv2ssd.core.config import PROJECT_ROOT
import tf2onnx

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a MobileNetV2 SSD model to ONNX.")
    parser.add_argument('--deploy_config', type=str, required=True, help='Path to the deployment configuration file.')
    
    # Creating the args dictionary
    args = parser.parse_args()
    
    return {
        'deploy_config': Path(args.deploy_config),
    }
    
def execute_convert():
    try:
        args= parse_args()
        
        # Config Parameters
        deploy_config = load_deploy_config(args['deploy_config'])
        model_save_path = PROJECT_ROOT / deploy_config['deploy']['saved_model_path']
        onnx_path = PROJECT_ROOT / deploy_config['deploy']['onnx_path']
        opset = deploy_config['deploy']['runtime']['opset']
        
        # Running the conversion process
        subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(model_save_path), "--output", str(onnx_path), "--opset", str(opset)], check=True)
        
        print(f"Converted → {onnx_path}")
        return 0
    
    except Exception as err:
        traceback.print_exc()
        return 1
    
if __name__ == '__main__':
    sys.exit(execute_convert())