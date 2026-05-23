import argparse
import subprocess
from pathlib import Path

import tensorflow as tf
import numpy as np

def parse_args():
   parser = argparse.ArgumentParser(description="Model exporter")
   
   parser.add_argument("--s3_path", required= True, help="S3 Path for the best checkpoint for the model", type=str)
   parser.add_argument("--config_path", required= True, help="Config directory path")
   parser.add_argument('--config_root', required= False, default='configs', help="Root directory for all config files")
   parser.add_argument('--output_dir',required= False, default='exported_model', help="Output directory to store the exported model")
   
   return parser.parse_args()

def main():
    
    arguments = parse_args()
    
    # Downloading checkpoint from the S3
    local_checkpoint = Path('/tmp/export_ckpt/best')
    local_checkpoint.mkdir(parents= True, exist_ok= True)
    
    export_path = Path(arguments.output_dir)
    config_path = Path(arguments.config_path)
    config_root = Path(arguments.config_root)
    
    subprocess.run(args=["aws","s3","sync",arguments.s3_path, str(local_checkpoint)], check= True)
    
    index_files = list(local_checkpoint.glob("ckpt-*.index"))
    if not index_files:
        raise FileNotFoundError(
            f"No checkpoint files found in {local_checkpoint}."
            f"Check S3_PATH is correct."
        )
        
    # Loading the config for the model
    from mobilenetv2ssd.core.config import load_config
    config = load_config(experiment_path=config_path, config_root=config_root)
    
    # Building the priors for the model
    from mobilenetv2ssd.models.ssd.orchestration.priors_orch import build_priors_from_config
    priors, priors_meta = build_priors_from_config(model_config= config)
    anchors_per_layer = priors_meta['anchors_per_cell'].numpy()
    
    # Building the model
    from mobilenetv2ssd.models.factory import build_ssd_model
    model = build_ssd_model(config= config, anchors_per_layer= anchors_per_layer)
    
    # Build the EMA
    from training.ema import build_ema
    ema = build_ema(config= config, model= model)
    
    # Restoring the checkpoint
    ckpt = tf.train.Checkpoint(model=model, ema=ema)
    ckpt_manager = tf.train.CheckpointManager(ckpt, str(local_checkpoint), max_to_keep= None)
    
    restore_path = ckpt_manager.latest_checkpoint
    if restore_path is None:
        latest_index = max(index_files, key=lambda path: int(path.stem.split("-")[1]))
        restore_path = str(latest_index.with_suffix(""))
        
    ckpt.restore(restore_path).expect_partial()
    
    # Exporting the saved model with EMA weights applied to it
    export_path.mkdir(parents= True, exist_ok= True)
    saved_model_path = export_path / "saved_model"
    
    H,W = config['input_size'][0], config['input_size'][1]
    input_spec = tf.TensorSpec([None,H,W,3], tf.float32, name="input_image")
    
    @tf.function(input_signature=[input_spec])
    def serve(x):
        offsets, logits = model(x, training= False)
        return {"offsets": offsets, "logits": logits}
    
    with ema.eval_context(model):
        tf.saved_model.save(model, str(saved_model_path), signatures={'serving_default': serve})
        
    priors_path = export_path / "priors_cxcywh.npy"
    np.save(str(priors_path), priors.numpy())
    

if __name__ == '__main__':
    main()

