from pathlib import Path
import yaml

def load_deploy_config(path: Path):
    
    with open(path) as file:
        return yaml.safe_load(file)