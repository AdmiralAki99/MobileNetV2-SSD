from fastapi import APIRouter
import boto3
import requests
from sqlalchemy import create_engine, text
from ..config import AIRFLOW_DB_URL

router = APIRouter()

INSTANCE_SPECS = {
    "g4dn.2xlarge": {"cpu": 8,  "memory_gb": 32},
    "g5.xlarge":    {"cpu": 4,  "memory_gb": 16},
}


@router.get("/airflow")
def get_airflow():
    engine = create_engine(AIRFLOW_DB_URL)
    with engine.connect() as conn:
        run = conn.execute(text("""
                SELECT run_id, state, start_date, end_date
                FROM dag_run where dag_id = 'etl_pipeline'
                ORDER BY start_date DESC LIMIT 1               
            """)).mappings().first()
        
        tasks = []
        if run:
            tasks = [dict(r) for r in conn.execute(text("""
                        SELECT task_id, state, duration, start_date
                        FROM task_instance
                        WHERE  dag_id = 'etl_pipeline' AND run_id = :rid
                        ORDER BY start_date                                    
                    """),{"rid": run["run_id"]}).mappings().all()]
            
        return {
            "dag_id": "etl_pipeline",
            "schedule": "0 2 * * *",
            "last_run": dict(run) if run else {},
            "tasks": tasks
        }
        
@router.get("/airflow/runs")
def get_airflow_runs():
    engine = create_engine(AIRFLOW_DB_URL)
    with engine.connect() as conn:
        runs = conn.execute(text("""
            SELECT run_id, state, run_type, start_date, end_date,
                   EXTRACT(EPOCH FROM (end_date - start_date)) AS duration
            FROM dag_run
            WHERE dag_id = 'etl_pipeline'
            ORDER BY start_date DESC
            LIMIT 50
        """)).mappings().all()
    return [dict(r) for r in runs]

@router.get("/airflow/runs/{run_id}/tasks")
def get_run_tasks(run_id: str):
    engine = create_engine(AIRFLOW_DB_URL)
    with engine.connect() as conn:
        tasks = conn.execute(text("""
            SELECT task_id, state, duration, start_date, end_date, try_number
            FROM task_instance
            WHERE dag_id = 'etl_pipeline' AND run_id = :rid
            ORDER BY start_date
        """), {"rid": run_id}).mappings().all()
    return [dict(r) for r in tasks]

@router.get("/ray")
def get_ray():
    ec2 = boto3.client("ec2", region_name="us-east-1")
    resp = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": ["etl-ray-worker"]},
        {"Name": "instance-state-name", "Values": ["running", "pending"]},
    ])
    reservations = resp["Reservations"]

    if not reservations:
        return {"status": "stopped", "nodes": [], "resources": {}}

    inst = reservations[0]["Instances"][0]
    ip = inst.get("PublicIpAddress", "")
    itype = inst.get("InstanceType", "")
    state = inst["State"]["Name"]
    specs = INSTANCE_SPECS.get(itype, {"cpu": 0, "memory_gb": 0})

    return {
        "status": "running" if state == "running" else "starting",
        "dashboard_url": f"http://{ip}:8265" if ip else "",
        "resources": {
            "cpu_total": specs["cpu"],
            "memory_total_gb": specs["memory_gb"],
        },
        "nodes": [{
            "id": inst["InstanceId"],
            "ip": ip,
            "status": "alive",
            "instance_type": itype,
        }],
    }
