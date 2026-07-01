import argparse
from mobilenetv2ssd.core.config import load_config
from ....config import CONFIGS_DIR
from .runner import run_tfrecord_job

def parse_args():
    parser = argparse.ArgumentParser(description="Run TFRecord Conversion job for datasets")
    parser.add_argument("--config", help="Config file directory", required=True, type=str)
    parser.add_argument("--stats-only", action="store_true", help="Compute and store box stats only; skip writing/uploading TFRecord shards")
    
    args= parser.parse_args()
    
    return {"config_path":args.config, "stats_only": args.stats_only}

def execute_tfrecords():
    args = parse_args()
    config = load_config(experiment_path=args["config_path"], config_root=str(CONFIGS_DIR))
    run_tfrecord_job(config=config, config_path=args["config_path"], stats_only = args['stats_only'])
    
    
if __name__ == '__main__':
    execute_tfrecords()