import onnxruntime as ort
import sys
import numpy as np
import json
import argparse
import traceback
from pathlib import Path
from typing import Any

from deploy import load_deploy_config
from mobilenetv2ssd.core.config import PROJECT_ROOT

def parse_args():
    parser = argparse.ArgumentParser(description="Validate a MobileNetV2 SSD ONNX model.")
    parser.add_argument('--deploy_config', type=str, required=True, help='Path to the deployment configuration file.')
    parser.add_argument('--print_config', action='store_true', help='Print the deployment config.')
    
    # Creating the args dictionary
    args = parser.parse_args()
    
    return {
        'deploy_config': Path(args.deploy_config),
        'print_config': args.print_config,
    }
    
    
def validate_onnx(onnx_path: Path, deploy_config: dict[str, Any]):
    # Creating an ingerence cession on the model
    onnx_session = ort.InferenceSession(str(onnx_path))
    
    # Printing out the inputs and outputs
    print("Inputs:",  [(input_.name, input_.shape) for input_ in onnx_session.get_inputs()])
    print("Outputs:", [(output_.name, output_.shape) for output_ in onnx_session.get_outputs()])
    
    # Getting info from the config
    H, W = deploy_config['deploy']['input']['size'][0], deploy_config['deploy']['input']['size'][1]
    num_classes = deploy_config['deploy']['classes']['num_classes']
    B = deploy_config['deploy']['runtime']['batch_size']
    
    dummy_image = np.zeros([B,H,W,3], dtype= np.float32)
    
    input_name = onnx_session.get_inputs()[0].name
    
    # Runnning the model
    raw_outputs = onnx_session.run(None, {input_name: dummy_image})
    
    outputs = onnx_session.get_outputs()
    
    results = {o.name: raw_outputs[i] for i, o in enumerate(outputs)}
    
    boxes = results['boxes']
    scores = results['scores']
    number_of_anchors = scores.shape[1]
    
    assert boxes.shape == (B, number_of_anchors, 4), f"Bad boxes shape: {boxes.shape}"
    assert scores.shape == (B, number_of_anchors, num_classes), f"Bad scores shape: {scores.shape}"
    
    print("PASS")
    
def execute_validate():
    try:
        
        # Get the args
        args = parse_args()
    
        # Read the config
        deploy_config = load_deploy_config(args['deploy_config'])
        
        if args['print_config']:
            print(json.dumps(deploy_config, indent= 2))
            return 0
    
        # Resolving the ONNX path
        onnx_path = PROJECT_ROOT / deploy_config['deploy']['onnx_path']
    
        # Calling the validate function
        validate_onnx(onnx_path= onnx_path, deploy_config= deploy_config)
        
        return 0
    except Exception as err:
        print("FAIL")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(execute_validate())