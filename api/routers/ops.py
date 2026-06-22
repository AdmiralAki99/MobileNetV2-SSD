from fastapi import APIRouter
import boto3
from sqlalchemy import create_engine, text
from ..config import AIRFLOW_DB_URL

router = APIRouter()

INSTANCE_SPECS = {
    "g4dn.2xlarge": {"cpu": 8, "memory_gb": 32},
    "g5.xlarge": {"cpu": 4, "memory_gb": 16},
}

KNOWN_DAGS = {
    "etl_pipeline":      {"label": "ETL Pipeline",      "schedule": "0 2 * * *"},
    "training_pipeline": {"label": "Training Pipeline", "schedule": "manual"},
}


@router.get("/dags")
def get_dags():
    return [{"dag_id": k, **v} for k, v in KNOWN_DAGS.items()]


@router.get("/airflow")
def get_airflow(dag_id: str = "etl_pipeline"):
    engine = create_engine(AIRFLOW_DB_URL)
    with engine.connect() as conn:
        run = conn.execute(text("""
                SELECT run_id, state, start_date, end_date
                FROM dag_run where dag_id = :dag_id
                ORDER BY start_date DESC LIMIT 1
            """), {"dag_id": dag_id}).mappings().first()

        tasks = []
        if run:
            tasks = [
                dict(r)
                for r in conn.execute(
                    text("""
                        SELECT task_id, state, duration, start_date
                        FROM task_instance
                        WHERE  dag_id = :dag_id AND run_id = :rid
                        ORDER BY start_date
                    """),
                    {"dag_id": dag_id, "rid": run["run_id"]},
                )
                .mappings()
                .all()
            ]

        schedule = KNOWN_DAGS.get(dag_id, {}).get("schedule", "—")
        return {"dag_id": dag_id, "schedule": schedule, "last_run": dict(run) if run else {}, "tasks": tasks}


@router.get("/airflow/runs")
def get_airflow_runs(dag_id: str = "etl_pipeline"):
    engine = create_engine(AIRFLOW_DB_URL)
    with engine.connect() as conn:
        runs = conn.execute(text("""
            SELECT run_id, state, run_type, start_date, end_date,
                   EXTRACT(EPOCH FROM (end_date - start_date)) AS duration
            FROM dag_run
            WHERE dag_id = :dag_id
            ORDER BY start_date DESC
            LIMIT 50
        """), {"dag_id": dag_id}).mappings().all()
    return [dict(r) for r in runs]


@router.get("/airflow/runs/{run_id}/tasks")
def get_run_tasks(run_id: str, dag_id: str = "etl_pipeline"):
    engine = create_engine(AIRFLOW_DB_URL)
    with engine.connect() as conn:
        tasks = (
            conn.execute(
                text("""
            SELECT task_id, state, duration, start_date, end_date, try_number
            FROM task_instance
            WHERE dag_id = :dag_id AND run_id = :rid
            ORDER BY start_date
        """),
                {"dag_id": dag_id, "rid": run_id},
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in tasks]


@router.get("/ray")
def get_ray():
    ec2 = boto3.client("ec2", region_name="us-east-1")
    resp = ec2.describe_instances(
        Filters=[
            {"Name": "tag:Name", "Values": ["etl-ray-worker"]},
            {"Name": "instance-state-name", "Values": ["running", "pending"]},
        ]
    )
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
        "nodes": [
            {
                "id": inst["InstanceId"],
                "ip": ip,
                "status": "alive",
                "instance_type": itype,
            }
        ],
    }
