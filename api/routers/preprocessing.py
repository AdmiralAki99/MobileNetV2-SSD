import os
import requests
from fastapi import APIRouter
from pydantic import BaseModel

AIRFLOW_URL  = os.getenv("AIRFLOW_URL", "http://airflow:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASSWORD", "")

router = APIRouter()

class TFRecordLaunchRequest(BaseModel):
    config_path: str
    
@router.post("/tfrecords/launch")
def launch_tfrecords(req: TFRecordLaunchRequest):
    response = requests.post(
        f"{AIRFLOW_URL}/api/v1/dags/tfrecord_pipeline/dagRuns",
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        json={"conf": {"config_path": req.config_path}},
    )
    response.raise_for_status()
    return {
        "status": 200,
        "dag_run_id": response.json()["dag_run_id"],
    }