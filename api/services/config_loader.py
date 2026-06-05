from ..config import CONFIGS_DIR

from mobilenetv2ssd.core.config import load_config

# Global directory macro for the config files


def list_experiment_configs():
    # Looking at all the YAML files
    experiments_directory = CONFIGS_DIR / "experiments"
    experiments = [file for file in experiments_directory.glob("*.yaml") if not file.name.startswith("_")]
    experiment_tracker = []
    for experiment in experiments:
        try:
            # Read the config file
            config = load_config(experiment, config_root=str(CONFIGS_DIR))
            experiment_tracker.append(
                {
                    "filename": experiment.name,
                    "experiment_id": config.get("experiment", {}).get("id"),
                    "description": config.get("experiment", {}).get("description", ""),
                    "instance_type": config.get("infrastructure", {}).get("instance_type"),
                }
            )
        except Exception:
            pass
    return experiment_tracker
