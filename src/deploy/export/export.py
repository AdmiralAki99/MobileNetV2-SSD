from pathlib import Path
import subprocess
import argparse
from typing import Any
import tensorflow as tf
import numpy as np
import json
import traceback
import sys

from mobilenetv2ssd.core.config import load_config, PROJECT_ROOT
from mobilenetv2ssd.models.ssd.orchestration.priors_orch import build_priors_from_config
from mobilenetv2ssd.models.factory import build_ssd_model
from training.ema import build_ema
from deploy import load_deploy_config

def parse_args():
    parser = argparse.ArgumentParser(description="Export a MobileNetV2 SSD model.")
    parser.add_argument('--deploy_config', type=str, required=True, help='Path to the deployment configuration file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint directory (local path or s3:// URI).')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to write saved_model/ and priors_cxcywh.npy. Overrides deploy config paths.')
    parser.add_argument('--print_config', action='store_true', help='Print the deployment config.')

    # Creating the args dictionary
    args = parser.parse_args()

    return {
        'deploy_config': Path(args.deploy_config),
        'checkpoint': args.checkpoint,
        'output_dir': Path(args.output_dir) if args.output_dir else None,
        'print_config': args.print_config,
    }
    
    
def download_checkpoint(checkpoint_path: str):
    if checkpoint_path.startswith("s3://"):
        # Need to download the checkpoint from S3 to a local path
        s3_relative = checkpoint_path.split("://", 1)[1]          # bucket/runs/...
        s3_relative = s3_relative.split("/", 1)[1]                 # runs/...
        local_path = PROJECT_ROOT / "checkpoints" / "s3" / s3_relative.strip("/")
        local_path.mkdir(parents=True, exist_ok=True)
        aws_bin = str(Path(sys.executable).parent / "aws")
        subprocess.run([aws_bin, "s3", "sync", checkpoint_path, str(local_path)], check=True)
        return local_path

    return Path(checkpoint_path) # If it is local there is no need for any downloading

def build_serve_model(model: tf.keras.Model, priors_np: np.ndarray, deploy_config: dict[str, Any]):
    # Need to build the model that needs to be served
    input_options = deploy_config['deploy']['input'] # This entire file needs to be there, no sentinel values
    variances = deploy_config['deploy']['priors']['variances']
    H, W, C = input_options['size']
    
    # Making the stuff to Bake into the model
    MEAN = tf.constant(input_options['mean'], dtype= tf.float32)
    STD = tf.constant(input_options['std'], dtype= tf.float32)
    PRIORS = tf.constant(priors_np, dtype=tf.float32)
    VAR_C = tf.constant(variances[0], dtype= tf.float32)
    VAR_S = tf.constant(variances[1], dtype= tf.float32)
    
    # Creating that tf-graph compiled function
    @tf.function(input_signature= [tf.TensorSpec([None, H, W, C], dtype= tf.float32, name="input_image")])
    def serve_model(x: tf.Tensor):
        # Doing the pipeline of:
        # 1. Normalizing
        # 2. Forward Pass
        # 3. Deocode Boxes
        # 4. Softmax Logits
        
        # Normalizing boxes
        x = (x - MEAN) / STD
        
        # Forward Pass
        predicted_offsets, predicted_logits = model(x, training= False)
        
        # Decode Boxes from cxcywh into xyxy
        
        tx, ty, tw, th = tf.split(predicted_offsets, num_or_size_splits= 4, axis= -1)
        cx, cy, w, h = tf.split(PRIORS, num_or_size_splits= 4, axis= -1)
        
        cx_decoded  = cx + tx * VAR_C * w
        cy_decoded = cy + ty * VAR_C * h
        w_decoded = w * tf.math.exp(tw * VAR_S)
        h_decoded = h * tf.math.exp(th * VAR_S)
        
        x1 = cx_decoded - w_decoded / 2
        y1 = cy_decoded - h_decoded / 2
        x2 = cx_decoded + w_decoded / 2
        y2 = cy_decoded + h_decoded / 2
        
        # Stacking everything together
        bboxes = tf.concat([x1,y1,x2,y2], axis= -1)
        
        # Softmax the raw logits
        scores = tf.nn.softmax(predicted_logits, axis= -1)
        
        # Return the info from the model
        return {
            'boxes': bboxes,
            'scores': scores
        }
        
    # Returning the compiled function for usage in C++
    return serve_model


def execute_export():
    # The entire export pipeline so that an ONNX model can be created
    try:
        args= parse_args()

        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        
        deploy_config = load_deploy_config(args['deploy_config'])
        
        # Checking if this is a print config run (info run)
        if args['print_config']:
            # Need to print the config and leave
            print(json.dumps(deploy_config, indent= 2))
            return 1
        
        # Now loading in the experiment path for the deploy config
        experiment_path = PROJECT_ROOT / deploy_config['experiment_path']
        experiment_config = load_config(experiment_path= experiment_path, config_root= PROJECT_ROOT/ "configs")
        
        priors, priors_meta = build_priors_from_config(model_config= experiment_config)
        number_anchors_per_layer = priors_meta['anchors_per_cell'].numpy()
        model = build_ssd_model(config= experiment_config, anchors_per_layer= number_anchors_per_layer)
        ema = build_ema(config= experiment_config, model= model)
        
        # Now need to download that checkpoint
        checkpoint_path = download_checkpoint(args['checkpoint'])
        index_files = list(checkpoint_path.glob("ckpt-*.index")) # Getting all the checkpoint paths from the directory
        if not index_files:
            raise ValueError(f"No checkpoint files in {checkpoint_path}")
        
        # Restoring the models weights using EMA or not
        chkpt = tf.train.Checkpoint(model= model, ema= ema)
        chkpt_manager = tf.train.CheckpointManager(chkpt, str(checkpoint_path), max_to_keep = None) # Loading the variables
        restore_path = chkpt_manager.latest_checkpoint
        if restore_path is None:
            # Need to find the latest checkpoint to load in
            latest = max(index_files, key=lambda file: int(file.stem.split("-")[1]))
            restore_path = str(latest.with_suffix(""))
            
        chkpt.restore(restore_path).expect_partial()
        
        # Saving the model
        if args['output_dir']:
            model_save_path = args['output_dir'] / "saved_model"
        else:
            model_save_path = PROJECT_ROOT / deploy_config['deploy']['saved_model_path']
        model_save_path.mkdir(parents= True, exist_ok= True)

        # Building the model with the stuff baked in
        serve = build_serve_model(model= model, priors_np= priors.numpy(), deploy_config= deploy_config)

        # Now loading in those weights
        with ema.eval_context(model= model):
            # Saving the mdoel
            tf.saved_model.save(model, export_dir= str(model_save_path), signatures= {"serving_default": serve})

        # Saving the priors too
        priors_path = model_save_path.parent / "priors_cxcywh.npy"
        np.save(str(priors_path), priors.numpy())
        
        # Now doing a last check to see if everything worked
        H, W = experiment_config['input_size'][0], experiment_config['input_size'][1]
        loaded_model = tf.saved_model.load(str(model_save_path))
        inference_fn = loaded_model.signatures["serving_default"] # Getting that inference function created before
        dummy_image= tf.zeros([1, H, W, 3], dtype= tf.float32)
        result = inference_fn(input_image= dummy_image)
        print(f"boxes: {tuple(result['boxes'].shape)}")
        print(f"scores: {tuple(result['scores'].shape)}")
        
        return 0 # Everything worked
    except Exception as err:
        traceback.print_exc()
        return 1
    
    
if __name__ == '__main__':
    sys.exit(execute_export()) # This thing prefers int returns
 