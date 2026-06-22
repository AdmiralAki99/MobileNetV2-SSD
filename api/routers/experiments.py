import re
from fastapi import APIRouter, HTTPException
from decimal import Decimal
from pydantic import BaseModel

import yaml

from ..config import CONFIGS_DIR
from ..services.ledger import get_ledger
from ..services.experiments import register_experiments, preview_experiment
from ..services.config_library import sync_config_library, upload_config

class RegisterRequest(BaseModel):
    config: dict
    task_type: str = "detector"
    git_commit: str | None = None

class SaveConfigRequest(BaseModel):
    category: str 
    name: str
    content_yaml: str

# Creating the router to use in the api server

router = APIRouter()


def _to_json(item: dict):
    json_dict = {}
    for obj in item:
        if isinstance(item[obj], Decimal):
            json_dict[obj] = float(item[obj])
        elif isinstance(item[obj], set):
            json_dict[obj] = list(item[obj])
        else:
            json_dict[obj] = item[obj]

    return json_dict


@router.get("")
def get_experiments():
    # Need to get the ledger and then the experiments from it
    experiment_ledger = get_ledger()
    experiments = experiment_ledger.list_experiments()
    # Now need to make the decimals floats to make the serializable
    return [_to_json(experiment) for experiment in experiments]


@router.post("/{experiment_id}/{fingerprint}/reset")
def reset_experiment(experiment_id: str, fingerprint: str):
    # Need to get the ledger and then the reset from it
    experiment_ledger = get_ledger()
    reset_counter = experiment_ledger.reset_failed(experiment_id=experiment_id)

    if reset_counter == 0:
        raise HTTPException(status_code=400, detail="No failed runs to reset")
    else:
        return {"status": 200, "Message": f"Reset Complete! Took {reset_counter} tries"}
    
@router.post("/register")
def register(req: RegisterRequest):
    return register_experiments(req.config, req.task_type, req.git_commit)

@router.get("/config-library")
def get_config_library():
    library: dict = {}

    base_dir = CONFIGS_DIR / "base"
    if base_dir.exists():
        for yaml_file in base_dir.rglob("*.yaml"):
            category = yaml_file.parent.name
            library.setdefault(category, []).append({
                "name": yaml_file.stem,
                "path": str(yaml_file.relative_to(CONFIGS_DIR)).replace("\\", "/"),
                "content": yaml.safe_load(yaml_file.read_text()),
            })

    data_dir = CONFIGS_DIR / "data"
    if data_dir.exists():
        for yaml_file in sorted(data_dir.glob("*.yaml")):
            library.setdefault("data", []).append({
                "name": yaml_file.stem,
                "path": str(yaml_file.relative_to(CONFIGS_DIR)).replace("\\", "/"),
                "content": yaml.safe_load(yaml_file.read_text()),
            })

    return library

@router.post("/config-library/refresh")
def refresh_library():
    sync_config_library()
    return {"status": "synced"}

@router.post("/config-library/save")
def save_config(req: SaveConfigRequest):
    name = re.sub(r"[^a-z0-9_\-]", "_", req.name.strip().lower())
    if not name:
        raise HTTPException(status_code=400, detail="Invalid config name")

    try:
        yaml.safe_load(req.content_yaml)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}")

    if req.category == "data":
        dest = CONFIGS_DIR / "data" / f"{name}.yaml"
    else:
        dest = CONFIGS_DIR / "base" / req.category / f"{name}.yaml"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(req.content_yaml, encoding="utf-8")

    relative = str(dest.relative_to(CONFIGS_DIR)).replace("\\", "/")
    try:
        upload_config(dest, relative)
    except Exception:
        pass  # S3 unavailable locally

    return {"path": relative, "name": name}

@router.post("/preview")
def preview(req: RegisterRequest):
    return preview_experiment(req.config, req.git_commit)
