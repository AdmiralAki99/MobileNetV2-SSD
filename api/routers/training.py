import os
import requests
import subprocess
from fastapi import APIRouter, HTTPException
from pathlib import Path
from pydantic import BaseModel

from ..services.ec2 import describe_instance, stop_training
from ..services.ledger import get_ledger
from ..config import EXPERIMENT_BUCKET, INFRA_DIR

AIRFLOW_URL  = os.getenv("AIRFLOW_URL", "http://airflow:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASSWORD", "")


class LaunchRequest(BaseModel):
    experiment_id: str
    fingerprint: str
    
class StopRequest(BaseModel):
    instance_id: str
    experiment_id: str
    fingerprint: str


router = APIRouter()

@router.post("/launch")
def launch_training(req: LaunchRequest):
    response = requests.post(
        f"{AIRFLOW_URL}/api/v1/dags/training_pipeline/dagRuns",
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        json={"conf":{"experiment_id": req.experiment_id, "fingerprint": req.fingerprint}}
    )
    
    response.raise_for_status()
    return {
        "status": 200,
        "dag_run_id": response.json()['dag_run_id']
    }
        


@router.post("/stop")
def stop_experiment(req: StopRequest):

    cmd_directory = INFRA_DIR
    try:
        # Stop the training container on the instance
        stop_training(req.instance_id)

        result = subprocess.run(
            args=["terraform", "destroy", "-auto-approve", "-target=aws_ec2_fleet.training"],
            cwd=cmd_directory,
            check=True,
            capture_output=True,
            text=True,
        )

        return {"status": 200, "message": f"Instance torn down: {result.stdout}"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)


@router.get("/{instance_id}/status")
def get_instance_status(instance_id: str):
    description = describe_instance(instance_id=instance_id)

    if description is None:
        raise HTTPException(status_code=404, detail="Instance does not exist")

    # Returning the description
    return {"status": 200, "message": description}
