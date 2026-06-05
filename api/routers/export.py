import os
import subprocess
from fastapi import APIRouter, HTTPException
import threading
import uuid
from ..services.s3 import check_artifacts
from ..config import PROJECT_ROOT, TF_PYTHON, ONNX_PYTHON
from pydantic import BaseModel


class ExportRequest(BaseModel):
    experiment_id: str
    fingerprint: str
    checkpoint_s3_path: str
    config_filename: str


class OnnxRequest(BaseModel):
    experiment_id: str


_jobs = {}


def _do_export(job_id, req: ExportRequest):
    # Running the export job
    try:
        subprocess.run(
            args=[
                TF_PYTHON,
                str(PROJECT_ROOT / "api" / "run_export.py"),
                "--s3_path",
                req.checkpoint_s3_path,
                "--config_path",
                str(PROJECT_ROOT / "configs" / "experiments" / req.config_filename),
            ],
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        _jobs[job_id]["status"] = "success"
    except subprocess.CalledProcessError as e:
        _jobs[job_id] = {"status": "error", "log": [str(e)]}


def _do_onnx(job_id, req: OnnxRequest):
    try:
        subprocess.run(
            args=[
                ONNX_PYTHON,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                str(PROJECT_ROOT / "exported_model" / "saved_model"),
                "--output",
                str(PROJECT_ROOT / "exported_model" / "model.onnx"),
                "--opset",
                "17",
            ],
            check=True,
        )
        _jobs[job_id]["status"] = "success"
    except subprocess.CalledProcessError as e:
        _jobs[job_id] = {"status": "error", "log": [str(e)]}


router = APIRouter()


@router.post("/savedmodel")
def save_model(req: ExportRequest):
    # Generating a job id
    job_id = f"export_{req.experiment_id}_{req.fingerprint}_{str(uuid.uuid4())[:8]}_{req.config_filename}"

    # Starting the job
    job_thread = threading.Thread(target=_do_export, args=(job_id, req), daemon=True)
    _jobs[job_id] = {"status": "running"}
    job_thread.start()
    return {"job_id": job_id, "status": "running"}


@router.post("/onnx")
def onnx_model(req: OnnxRequest):
    # Generating a job id
    job_id = f"onnx_{req.experiment_id}_{str(uuid.uuid4())[:8]}"

    # Starting the job
    job_thread = threading.Thread(target=_do_onnx, args=(job_id, req), daemon=True)
    _jobs[job_id] = {"status": "running"}
    job_thread.start()

    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}")
def get_status(job_id: str):
    if job_id in _jobs:
        return {"status": 200, "job_status": _jobs[job_id]}
    else:
        raise HTTPException(status_code=404, detail="Job Id does not exist in the queue")


@router.get("/{experiment_id}/artifacts")
def get_artifacts(experiment_id):
    return {"status": 200, "artifact_status": check_artifacts(experiment_id=experiment_id)}
