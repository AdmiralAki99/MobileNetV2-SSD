import argparse
import os
import re
import yaml

from ..etl.runner import run_etl


def _expand_env_vars(value):
    if isinstance(value, str):
        def _replace(match):
            var, default = match.group(1), match.group(2) or ''
            return os.environ.get(var, default)
        return re.sub(r'\$\{([^}:]+)(?::-(.*?))?\}', _replace, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value

def parse_args():
    parser = argparse.ArgumentParser(description="Run ETL jobs for videos")
    parser.add_argument('--config', help="Config file directory", required= True,type=str)
    parser.add_argument('--videos', nargs="+", help="Directory for video files", required= True, type= str)
    
    args = parser.parse_args()
    
    return {
        'config_path': args.config,
        'videos_path': args.videos
    }
    
def execute_etl():
    args = parse_args()
    with open(args['config_path']) as file:
        config = _expand_env_vars(yaml.safe_load(file))

    run_etl(config=config['etl'], video_paths=args['videos_path'], config_path=args['config_path'])
    
    
if __name__ == "__main__":
    execute_etl()

    