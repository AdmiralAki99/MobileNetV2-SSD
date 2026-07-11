from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowException, AirflowFailException
from airflow.utils.email import send_email

import os
from datetime import datetime, timedelta, timezone
import boto3
from botocore.exceptions import ClientError
import subprocess
import time
import requests

# Constants for the DAG 
INFRA_DIR = "/app/infrastructure"
TABLE = "ml-experiment-ledger"
REGION = "us-east-1"
EXPERIMENT_BUCKET = "akhilesh-ml-experiments"

OWNER_EMAIL = os.environ.get("AIRFLOW_OWNER_EMAIL", "")
LOCAL_MODE = os.environ.get("TRAINING_LOCAL_MODE", "false").lower() == "true"
PROMOTION_MIN_MAP = float(os.environ.get("PROMOTION_MIN_MAP", "0.70"))
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
DEPLOY_CONFIG = os.environ.get("DEPLOY_CONFIG", "configs/deploy/mobilenetv2_ssd_voc_jetson.yaml")
CALIBRATION_DIR = os.environ.get("CALIBRATION_DIR", "")
ONNX_CONVERTER_IMAGE = os.environ.get("ONNX_CONVERTER_IMAGE", "mobilenetv2-ssd_onnx_converter")

default_args = {
    "owner": os.environ.get("AIRFLOW_OWNER", "airflow"),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email": [OWNER_EMAIL] if OWNER_EMAIL else [],
    "email_on_failure": bool(OWNER_EMAIL),
}

def _ledger():
    return boto3.resource("dynamodb", region_name=REGION).Table(TABLE)

# Check if experiment exsits in the DynamoDB ledger
def _check_experiment(**context):
    conf = context['dag_run'].conf or {}
    experiment_id = conf.get("experiment_id")
    fingerprint = conf.get("fingerprint")
    if not experiment_id or not fingerprint:
        raise AirflowException("trigger must include the experiment_id and fingerprint")
    
    row = _ledger().get_item(Key={"experiment_id":experiment_id, "fingerprint": fingerprint}).get("Item")
    if not row:
        raise AirflowException("Experiment not found")
    if row.get("status") != "pending":
        raise AirflowException(f"Experiment is '{row.get("status")}', not pending")
    
    return {
        "experiment_id": experiment_id,
        "fingerprint": fingerprint,
        "config_ref": row['config_ref'],
        "use_tfrecords": str(bool(row.get("use_tfrecords", False))).lower(),
        "instance_type": row.get("instance_type", "g4dn.2xlarge")
    }
    
# Launching the training job
def _launch_training_job(**context):
    row = context['ti'].xcom_pull(task_ids="check_experiment")
    config_uri = f"s3://{EXPERIMENT_BUCKET}/{row['config_ref']}"
    subprocess.run(["terraform","workspace","select","default"], cwd= INFRA_DIR, check= True, capture_output= True, text= True)
    subprocess.run(["terraform",
                    "apply",
                    "-auto-approve",
                    "-target=aws_ec2_fleet.training",
                    "-var", f"experiment_config={config_uri}",
                    "-var", f"use_tfrecords={row['use_tfrecords']}",
                    "-var", f'instance_types=["{row["instance_type"]}"]'],
                    cwd= INFRA_DIR, check= True, capture_output= True, text= True
    )
    
    
def _poll_training_success(**context):
    row = context['ti'].xcom_pull(task_ids="check_experiment")
    key = {"experiment_id": row["experiment_id"], "fingerprint": row["fingerprint"]}
    status = _ledger().get_item(Key=key).get("Item", {}).get("status")
    if status == "failed":
        raise AirflowFailException("Training failed")
    return status == "success"


def _email_report(**context):
    if not OWNER_EMAIL:
        return
    
    row = context["ti"].xcom_pull(task_ids="check_experiment")
    item = _ledger().get_item(Key={"experiment_id": row["experiment_id"],"fingerprint": row["fingerprint"]}).get("Item", {})
    html = f"""<h2>Training — {row['experiment_id']} ({row['fingerprint']})</h2>
    <p>Status: {item.get('status')}</p>
    <p>Best metric: {item.get('best_metric','-')}</p>
    <p>Steps: {item.get('total_steps','-')}</p>
    <p>Checkpoint: {item.get('checkpoint_s3_path','-')}</p>"""
    if item.get("failure_reason"):
        html += f"<p>Failure: {item['failure_reason']}</p>"
    send_email(to=OWNER_EMAIL, subject=f"Training {item.get('status','?')} — {row['experiment_id']}", html_content=html)
    
def _run_onnx_convert(**context):
    row = context["ti"].xcom_pull(task_ids="check_experiment")
    result = subprocess.run([
        "docker", "run", "--rm",
        "-e", f"AWS_ACCESS_KEY_ID={os.environ.get('AWS_ACCESS_KEY_ID','')}",
        "-e", f"AWS_SECRET_ACCESS_KEY={os.environ.get('AWS_SECRET_ACCESS_KEY','')}",
        "-e", f"AWS_DEFAULT_REGION={REGION}",
        "-v", f"{CALIBRATION_DIR}:/calibration",
        ONNX_CONVERTER_IMAGE,
        "--deploy_config", DEPLOY_CONFIG,
        "--calibration_dir", "/calibration",
        "--num_calibration", "100",
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise AirflowException(f"ONNX conversion failed:\n{result.stderr}")

def _gate_promotion(**context):
    row = context["ti"].xcom_pull(task_ids="check_experiment")
    key = {"experiment_id": row["experiment_id"], "fingerprint": row["fingerprint"]}
    item = _ledger().get_item(Key=key).get("Item", {})
    best_metric = float(item.get("best_metric", 0))
    status = "passed" if best_metric >= PROMOTION_MIN_MAP else "failed"
    _ledger().update_item(
        Key=key,
        UpdateExpression="SET promotion_status = :ps, promotion_threshold = :pt, promoted_at = :now",
        ExpressionAttributeValues={
            ":ps": status,
            ":pt": str(PROMOTION_MIN_MAP),
            ":now": datetime.now(timezone.utc).isoformat(),
        }
    )
    if status == "failed":
        raise AirflowException(f"Model did not pass promotion gate (mAP={best_metric:.4f} < {PROMOTION_MIN_MAP})")

def _notify_cd(**context):
    if not GITHUB_REPO or not GITHUB_TOKEN:
        return
    row = context["ti"].xcom_pull(task_ids="check_experiment")
    requests.post(
        f"https://api.github.com/repos/{GITHUB_REPO}/dispatches",
        headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
        json={"event_type": "model-promoted", "client_payload": {"experiment_id": row["experiment_id"], "fingerprint": row["fingerprint"]}},
    ).raise_for_status()

with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    schedule_interval= None,
    start_date= datetime(2026,1,1),
    catchup= False,
    max_active_runs = 1
) as dag:
    check_experiment = PythonOperator(
        task_id="check_experiment", 
        python_callable=_check_experiment
    )
    
    launch_training_job = PythonOperator(
        task_id="launch_training_job",
        python_callable= _launch_training_job
    )
    
    wait_for_training = PythonSensor(
        task_id="wait_for_completion",
        python_callable=_poll_training_success,
        mode="reschedule",
        poke_interval=120,
        timeout=24 * 3600,
    )
    
    teardown = BashOperator(
        task_id="teardown_ec2",
        trigger_rule="all_done",
        bash_command=f"cd {INFRA_DIR} && terraform workspace select default && terraform destroy -auto-approve -target=aws_ec2_fleet.training"
    )
    
    email_report = PythonOperator(
        task_id="email_report",
        python_callable=_email_report,
        trigger_rule="all_done"
    )
    
    onnx_convert = PythonOperator(
        task_id="onnx_convert",
        python_callable=_run_onnx_convert,
    )
    
    gate = PythonOperator(
        task_id="promotion_gate",
        python_callable=_gate_promotion,
    )
    
    notify_cd = PythonOperator(
        task_id="notify_cd",
        python_callable=_notify_cd,
        trigger_rule="all_success",
    )
    
    check_experiment >> launch_training_job >> wait_for_training >> teardown >> onnx_convert >> gate >> notify_cd >> email_report