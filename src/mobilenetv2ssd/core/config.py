from pathlib import Path
from copy import deepcopy
import re
import os
import yaml
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")

# Creating a main function to load the different files for the different configs for the components
def load_config(experiment_path: str | Path, config_root: str | Path | None = None, overrides: list[str] | None = None,):
    # Resolving the config root
    if config_root is None:
        config_root = PROJECT_ROOT / "configs"
    else:
        config_root = Path(config_root)
        
    # Resolving the experiment path
    exp_path = Path(experiment_path)
    if not exp_path.is_absolute() and not exp_path.exists():
        exp_path = config_root / exp_path
        
    experiment = read_yaml(exp_path)
    
    # Gettings the defaults and recipes
    defaults = experiment.get('defaults', {})
    recipes = experiment.get('recipes', {})
    
    # Merging all the configs together
    merged_config: dict[str, Any] = {}
    for component, default_path in defaults.items():
        # Use recipe if exists, otherwise use default
        config_path = recipes.get(component, default_path)
        full_path = config_root / config_path
        if full_path.exists():
            merged_config = merge_dict(merged_config, read_yaml(full_path))
            
    # Merging the experiment overrides
    exp_metadata_keys = {'experiment', 'infrastructure', 'defaults', 'recipes', 'overrides'}
    merged_config = merge_dict(merged_config, {key: value for key, value in experiment.items() if key not in exp_metadata_keys})
    
    if 'overrides' in experiment:
        merged_config = merge_dict(merged_config, experiment['overrides'])
    
    merged_config['experiment'] = experiment.get('experiment', {})
    merged_config['infrastructure'] = experiment.get('infrastructure', {})
    
    # Applying CLI overrides
    if overrides:
        merged_config = merge_dict(merged_config, parse_cli_overrides(overrides))
        
    merged_config = inject_env_vars(merged_config)
    merged_config = _resolve_paths(merged_config, PROJECT_ROOT)

    return merged_config
    

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def read_yaml(path: Path | str):
    # Checking if it is a string, if so convert to path
    if type(path) == str:
        path = Path(path)
        
    if not path.exists():
        return {}
        
    # Function to read YAML file
    with open(path, 'r') as file:
        data = yaml.load(file,Loader=yaml.SafeLoader)
        
    return data if data else {}

def merge_dict(base: dict[str , any], override: dict[str , any]) -> dict[str , any]:
    result = deepcopy(base)

    for key, override_value in override.items():
        if key in result:
            base_value = result[key]
            # If both are dicts, merge recursively
            if isinstance(base_value, dict) and isinstance(override_value, dict):
                result[key] = merge_dict(base_value, override_value)
            else:
                # Scalar / list / non-dict → override wins
                result[key] = deepcopy(override_value)
        else:
            # Key only in override → just add it
            result[key] = deepcopy(override_value)

    return result

def parse_cli_overrides(overrides: list[str]):
    result: dict[str, any] = {}
    
    for item in overrides:
        if not item:
            continue  # skip empty strings

        
        if "=" not in item:
            continue

        left, right = item.split("=", 1)
        left = left.strip()
        raw_value = right.strip()

        path = [part.strip() for part in left.split(".") if part.strip()]
        if not path:
            continue

        value = _parse_value(raw_value)

        # 4) Build a nested dict for this one override
        #    e.g. ["train", "batch_size"], 64 → {"train": {"batch_size": 64}}
        cur: dict[str, any] = {}
        d = cur
        for key in path[:-1]:
            d[key] = {}
            d = d[key]
        d[path[-1]] = value

        # 5) Merge this small nested dict into the global result
        result = merge_dict(result, cur)
        
    return result

def _parse_value(raw: str):
    text = raw.strip()

    # Booleans
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    # Int
    try:
        return int(text)
    except ValueError:
        pass

    # Float
    try:
        return float(text)
    except ValueError:
        pass

    return text

def inject_env_vars(cfg: dict[str, any]):
    return _inject_env_vars_obj(cfg)

def _inject_env_vars_obj(obj: any):

    # Dict → recurse on values
    if isinstance(obj, dict):
        return {k: _inject_env_vars_obj(v) for k, v in obj.items()}

    # List / tuple → recurse on elements
    if isinstance(obj, list):
        return [_inject_env_vars_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_inject_env_vars_obj(v) for v in obj)

    # String → substitute env placeholders
    if isinstance(obj, str):
        return _inject_env_vars_string(obj)

    # Anything else → leave as is
    return obj

def _inject_env_vars_string(s: str) -> str:
    """Apply ${VAR} / ${VAR:-default} substitution to a single string."""

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)   # VAR
        default = match.group(3)    # default (may be None)

        if var_name in os.environ:
            return os.environ[var_name]

        if default is not None:
            # ${VAR:-default} and VAR not set → use default
            return default

        # ${VAR} with no default and not set → hard error
        raise ValueError(
            f"Environment variable '{var_name}' is not set and no default "
            f"was provided in placeholder '{match.group(0)}'"
        )

    return _ENV_PATTERN.sub(replacer, s)

def _resolve_paths(cfg: dict[str, Any], project_root: Path):
    return _resolve_paths_obj(cfg, project_root)

def _resolve_paths_obj(obj: Any, project_root: Path):
    if isinstance(obj, dict):
        resolved: dict[str, Any] = {}
        for key, value in obj.items():
            if isinstance(value, str) and _is_path_key(key):
                resolved[key] = _resolve_single_path(value, project_root)
            else:
                resolved[key] = _resolve_paths_obj(value, project_root)
        return resolved
    
    if isinstance(obj, list):
        return [_resolve_paths_obj(v, project_root) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_resolve_paths_obj(v, project_root) for v in obj)
    
    return obj

def _resolve_single_path(path_str: str, project_root: Path):
    expanded = os.path.expanduser(path_str)
    
    if os.path.isabs(expanded):
        return expanded
    
    absolute = (project_root / expanded).resolve()
    
    return str(absolute)

def _is_path_key(key: str):
    key = key.lower()
    
    if key in {"root", "dir", "path", "runs_root", "output_dir", "classes_file"}:
        return True
    
    if key.endswith("_dir") or key.endswith("_root") or key.endswith("_path"):
        return True
    
    return False



# MACROS
PROJECT_ROOT = get_project_root()