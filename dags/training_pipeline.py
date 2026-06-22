from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowException, AirflowFailException
from airflow.utils.email import send_email

import os
from datetime import datetime, timedelta
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
    
    check_experiment >> launch_training_job >> wait_for_training >> teardown >> email_report
    