import subprocess
from fastapi import APIRouter, HTTPException
from pathlib import Path
from ..services.ec2 import describe_instance, stop_training

from pydantic import BaseModel


class LaunchRequest(BaseModel):
    experiment_id: str
    fingerprint: str
    config_filename: str


class StopRequest(BaseModel):
    instance_id: str
    experiment_id: str
    fingerprint: str


router = APIRouter()


@router.post("/launch")
def launch_training(req: LaunchRequest):

    cmd_directory = Path(__file__).parent.parent.parent / "infrastructure"
    try:
        # First trying to create a workspace for the experiment
        try:
            result = subprocess.run(
                args=["terraform", "workspace", "new", f"{req.experiment_id}_{req.fingerprint}"],
                cwd=cmd_directory,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            pass
        # Now selecting that workspace, if it already existed then the error is eaten
        result = subprocess.run(
            args=["terraform", "workspace", "select", f"{req.experiment_id}_{req.fingerprint}"],
            cwd=cmd_directory,
            check=True,
            capture_output=True,
            text=True,
        )

        result = subprocess.run(
            args=[
                "terraform",
                "apply",
                "-auto-approve",
                "-var",
                f"experiment_config=../configs/experiments/{req.config_filename}",
            ],
            cwd=cmd_directory,
            check=True,
            capture_output=True,
            text=True,
        )

        return {"status": 200, "message": f"Subprocess sucessfully executed with {result.stdout}!"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)


@router.post("/stop")
def stop_experiment(req: StopRequest):

    cmd_directory = Path(__file__).parent.parent.parent / "infrastructure"
    try:

        # Stopping the docker container from running
        stop_training(req.instance_id)

        # Now selecting that workspace
        result = subprocess.run(
            args=["terraform", "workspace", "select", f"{req.experiment_id}_{req.fingerprint}"],
            cwd=cmd_directory,
            check=True,
            capture_output=True,
            text=True,
        )

        # Deleting that workspace
        result = subprocess.run(
            args=["terraform", "destroy", "-auto-approve"],
            cwd=cmd_directory,
            check=True,
            capture_output=True,
            text=True,
        )

        # Swapping to the default values
        result = subprocess.run(
            args=["terraform", "workspace", "select", "default"],
            cwd=cmd_directory,
            check=True,
            capture_output=True,
            text=True,
        )

        result = subprocess.run(
            args=["terraform", "workspace", "delete", f"{req.experiment_id}_{req.fingerprint}"],
            cwd=cmd_directory,
            check=True,
            capture_output=True,
            text=True,
        )

        return {"status": 200, "message": f"Subprocess sucessfully executed with {result.stdout}!"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)


@router.get("/{instance_id}/status")
def get_instance_status(instance_id: str):
    description = describe_instance(instance_id=instance_id)

    if description is None:
        raise HTTPException(status_code=404, detail="Instance does not exist")

    # Returning the description
    return {"status": 200, "message": description}
