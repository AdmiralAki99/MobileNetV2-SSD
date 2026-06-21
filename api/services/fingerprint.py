from typing import Any
from mobilenetv2ssd.core.fingerprint import Fingerprinter
from mobilenetv2ssd.core.config import _is_path_key

FINGERPRINT_KEYS = ["input_size","num_classes","backbone","heads","priors","loss",
                    "sampler","matcher","augmentation","optimizer","scheduler","train","data","eval"]
FINGERPRINT_EXCLUDES = {
    "train": {"diagnostics"},
    "eval": {"interval_epochs", "visualization"},
    "data": {"loader", "root", "classes_file"},
}

def _strip_path_keys(obj):
    if isinstance(obj, dict):
        return {k: _strip_path_keys(v) for k, v in obj.items() if not _is_path_key(k)}
    if isinstance(obj, list):
        return [_strip_path_keys(v) for v in obj]
    return obj

def compute_fingerprint(config: dict[str, Any], git_commit: str | None = None):
    fp = {}
    for key in FINGERPRINT_KEYS:
        if key not in config:
            continue
        value = config[key]
        if key in FINGERPRINT_EXCLUDES and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in FINGERPRINT_EXCLUDES[key]}
        fp[key] = _strip_path_keys(value)
    if git_commit is not None:
        fp["git_commit"] = git_commit
    return Fingerprinter().fingerprint(fp)