from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.email import send_email

from datetime import datetime, timedelta
from urllib.parse import urlparse
import boto3
import os
import psycopg2
import requests
import time
import yaml
import subprocess

OWNER_EMAIL = os.environ.get("AIRFLOW_OWNER_EMAIL", "")
LOCAL_MODE = os.environ.get("ETL_LOCAL_MODE", "false").lower() == "true"

default_args = {
    "owner": os.environ.get("AIRFLOW_OWNER", "airflow"),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email": [OWNER_EMAIL] if OWNER_EMAIL else [],
    "email_on_failure": bool(OWNER_EMAIL),
}

def _run_tfrecord_job(**context):
    config_path = context["params"]["config_path"]   # e.g. configs/experiments/exp003_visdrone_run.yaml
    host_dir = os.environ.get("HOST_PROJECT_DIR", os.getcwd())
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "--network", "mobilenetv2-ssd_default",
            "-e", f"AWS_ACCESS_KEY_ID={os.environ.get('AWS_ACCESS_KEY_ID','')}",
            "-e", f"AWS_SECRET_ACCESS_KEY={os.environ.get('AWS_SECRET_ACCESS_KEY','')}",
            "-e", f"AWS_DEFAULT_REGION={os.environ.get('AWS_DEFAULT_REGION','us-east-1')}",
            "-e", f"ETL_DATABASE_URL={os.environ.get('ETL_DATABASE_URL','')}",
            "-v", f"{host_dir}/configs:/app/configs",
            "-v", f"{host_dir}/datasets:/app/datasets",
            "mobilenetv2-ssd-tfrecords:latest",
            "--config", config_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"TFRecord container failed:\n{result.stderr}")
    return {"status": "completed", "config": config_path}

def _send_summary_email(**context):
    if not OWNER_EMAIL:
        return
    config_path = context["params"]["config_path"]
    db_url = os.environ.get("ETL_DATABASE_URL") or "postgresql://user:password@postgres:5432/etl_db"
    parsed = urlparse(db_url)
    conn = psycopg2.connect(
        host=parsed.hostname, port=parsed.port,
        dbname=parsed.path.lstrip("/"),
        user=parsed.username, password=parsed.password,
    )
    with conn.cursor() as cur:
        # per-split totals
        cur.execute(
            "SELECT name, split, num_images, num_boxes "
            "FROM dataset WHERE config_path = %s ORDER BY split",
            (config_path,),
        )
        split_rows = cur.fetchall()
        # class distribution across both splits
        cur.execute(
            "SELECT b.class_label, COUNT(*) AS total "
            "FROM boxes b JOIN dataset d ON b.dataset_id = d.id "
            "WHERE d.config_path = %s GROUP BY b.class_label ORDER BY total DESC",
            (config_path,),
        )
        class_rows = cur.fetchall()
    conn.close()

    split_html = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>"
        for r in split_rows
    )
    class_html = "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td></tr>" for r in class_rows)
    html = f"""
    <h2>TFRecord Generation Summary</h2>
    <p><strong>Config:</strong> {config_path}</p>
    <table border="1" cellpadding="6" style="border-collapse:collapse">
      <thead><tr><th>Dataset</th><th>Split</th><th>Images</th><th>Boxes</th></tr></thead>
      <tbody>{split_html}</tbody>
    </table>
    <h3>Class Distribution</h3>
    <table border="1" cellpadding="6" style="border-collapse:collapse">
      <thead><tr><th>Class</th><th>Count</th></tr></thead>
      <tbody>{class_html}</tbody>
    </table>
    """
    send_email(to=OWNER_EMAIL, subject="TFRecord Generation Complete", html_content=html)


with DAG(
    dag_id="tfrecord_pipeline",
    default_args=default_args,
    schedule_interval=None,          # ← triggered on-demand by the API, not scheduled
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    params={"config_path": "configs/experiments/exp002_cloud_run.yaml"},
) as dag:
    run_tfrecords = PythonOperator(
        task_id="run_tfrecords",
        python_callable=_run_tfrecord_job,
    )
    
    send_summary = PythonOperator(
        task_id="email_summary",
        python_callable=_send_summary_email,
        trigger_rule="all_done",   # send even if generation failed, like etl
    )
    
    run_tfrecords >> send_summary