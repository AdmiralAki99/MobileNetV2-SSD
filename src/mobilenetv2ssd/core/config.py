from pathlib import Path
from copy import deepcopy
import re
import os
import yaml
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")

# Creating a main function to load the different files for the different configs for the components
def load_config(cfg_path: str | Path, model_cfg_path: str | Path | None = None, data_cfg_path: str | Path | None = None, eval_cfg_path: str | Path | None = None, overrides: list[str] | None = None,):
    # The arguments will hold 5 distinct things
    # 1. Main Config file (train file)
    # 2. Model Config File (Optional)
    # 3. Data Config File (Optional)
    # 4. Evaluation Config File (Optional)
    # 5. Override Options (Optional)
    
    # Read the training yaml file
    train_config  = read_yaml(PROJECT_ROOT / cfg_path)
    
    if model_cfg_path is None:
        model_cfg_path = PROJECT_ROOT / train_config["include"]["model_cfg"]
    else:
        model_cfg_path = PROJECT_ROOT / model_cfg_path

    if data_cfg_path is None:
        data_cfg_path = PROJECT_ROOT / train_config["include"]["data_cfg"]
    else:
        data_cfg_path = PROJECT_ROOT / data_cfg_path

    if eval_cfg_path is None:
        eval_cfg_path = PROJECT_ROOT / train_config["include"]["eval_cfg"]
    else:
        eval_cfg_path = PROJECT_ROOT / eval_cfg_path
    
    # Reading the included files that are wanted
    model_config = read_yaml(model_cfg_path)
    data_config  = read_yaml(data_cfg_path)
    eval_config  = read_yaml(eval_cfg_path)
    
    # Merging the dicts to have the main dict
    main_config = merge_dict(train_config,model_config)
    main_config = merge_dict(main_config,data_config)
    main_config = merge_dict(main_config,eval_config)
    
    # Now checking for optional arguments
    if overrides:
        override_dicts = parse_cli_overrides(overrides)
        main_config = merge_dict(main_config, override_dicts)
    
    main_config = inject_env_vars(main_config)
    
    main_config = _resolve_paths(main_config, PROJECT_ROOT)
    
    return main_config
    

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def read_yaml(path: Path | str):
    # Checking if it is a string, if so convert to path
    if type(path) == str:
        path = Path(path)
        
    # Function to read YAML file
    with open(path, 'r') as file:
        data = yaml.load(file,Loader=yaml.SafeLoader)
        
    return data

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