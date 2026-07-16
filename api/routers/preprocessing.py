import os
import json
import requests
import docker as docker_sdk
from fastapi import APIRouter
from fastapi import HTTPException, Query
from pydantic import BaseModel
import psycopg2

from ..config import CONFIGS_DIR

AIRFLOW_URL  = os.getenv("AIRFLOW_URL", "http://airflow:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASSWORD", "")

router = APIRouter()

class TFRecordLaunchRequest(BaseModel):
    config_path: str
    
class ClusterRequest(BaseModel):
    dataset: str
    split: str
    algorithm: str = "kmeans"
    num_aspect_ratios: int = 3
    priors: str
    out: str | None = None

def _ledger_conn():
    return psycopg2.connect(os.environ["ETL_DATABASE_URL"])

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
    
@router.get("/datasets")
def list_datasets():
    conn = _ledger_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT name, split, num_images, num_boxes, config_path, updated_at
                FROM dataset ORDER BY name, split
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return {
                "datasets": [
                                {
                                    "name": row[0], 
                                    "split": row[1], 
                                    "num_images": row[2], 
                                    "num_boxes": row[3],
                                    "config_path": row[4], 
                                    "updated_at": str(row[5])
                                } for row in rows
                            ]
        }
    
@router.get("/priors")
def list_priors():
    priors_dir = CONFIGS_DIR / "base" / "priors"
    return {"priors": sorted(p.name for p in priors_dir.glob("*.yaml"))}

@router.get("/ledger/box-sizes")
def box_sizes(dataset: str = Query(...), split: str = Query(...), limit: int = 5000):
    conn = _ledger_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT b.norm_width, b.norm_height
                FROM boxes b JOIN dataset d ON b.dataset_id = d.id
                WHERE d.name = %s AND d.split = %s
                ORDER BY random() LIMIT %s
            """, (dataset, split, limit))
            rows = cur.fetchall()
    finally:
        conn.close()
    return  {
                "dataset": dataset, "split": split,
                "points": [[float(w), float(h)] for w, h in rows]
            }

def _run_clustering(req, emit_json=True):
    host = os.environ.get("HOST_PROJECT_DIR", os.getcwd())
    try:
        client = docker_sdk.from_env()
        priors_in_container = f"/app/configs/base/priors/{req.priors}"
        cmd = [
            "--dataset", req.dataset, "--split", req.split,
            "--algorithm", req.algorithm,
            "--num-aspect-ratios", str(req.num_aspect_ratios),
            "--priors", priors_in_container,
        ]
        if req.out: cmd += ["--out", f"/app/configs/base/priors/{req.out}"]
        if emit_json: cmd += ["--json"]
        output = client.containers.run(
            "mobilenetv2-ssd-clustering:latest",
            command=cmd,
            remove=True,
            network="mobilenetv2-ssd_default",
            environment={"ETL_DATABASE_URL": os.environ.get("ETL_DATABASE_URL", "")},
            volumes={f"{host}/configs": {"bind": "/app/configs", "mode": "rw"}},
        )
        text = output.decode("utf-8").strip()
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            raise HTTPException(status_code=500, detail="no output from clustering container")
        return json.loads(lines[-1])
    except HTTPException:
        raise
    except docker_sdk.errors.ContainerError as e:
        raise HTTPException(status_code=500, detail=e.stderr.decode("utf-8") if e.stderr else str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clustering/derive")
def derive(req: ClusterRequest):
    req.out = None
    return {"status": 200, "result": _run_clustering(req)}

@router.post("/clustering/export")
def export(req: ClusterRequest):
    if not req.out:
        raise HTTPException(400, "`out` is required for export")
    return {"status": 200, "result": _run_clustering(req)}