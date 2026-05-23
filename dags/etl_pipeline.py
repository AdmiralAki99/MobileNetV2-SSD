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

OWNER_EMAIL = os.environ.get('AIRFLOW_OWNER_EMAIL', '')
LOCAL_MODE = os.environ.get('ETL_LOCAL_MODE', 'false').lower() == 'true'

default_args = {
    'owner': os.environ.get('AIRFLOW_OWNER', 'airflow'),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': [OWNER_EMAIL] if OWNER_EMAIL else [],
    'email_on_failure': bool(OWNER_EMAIL)
}


def _get_pending_videos(config_path: str) -> list:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    video_dir = config['etl']['videos']['input_dir']
    extensions = tuple(config['etl']['videos']['extensions'])
    if not os.path.exists(video_dir):
        return []
    all_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.lower().endswith(extensions)
    ]
    if not all_files:
        return []

    db_url = config['etl']['database']['url']
    parsed = urlparse(db_url)
    conn = psycopg2.connect(
        host=parsed.hostname, port=parsed.port,
        dbname=parsed.path.lstrip('/'),
        user=parsed.username, password=parsed.password
    )
    with conn.cursor() as cur:
        cur.execute("SELECT source_file FROM videos WHERE status = 'completed'")
        completed = {row[0] for row in cur.fetchall()}
    conn.close()
    return [f for f in all_files if f not in completed]


def _run_etl_job(**context):
    config_path = context['params']['config_path']
    video_paths = _get_pending_videos(config_path)

    if not video_paths:
        return {'skipped': True}

    if LOCAL_MODE:
        import subprocess
        host_dir = os.environ.get('HOST_PROJECT_DIR', os.getcwd())
        result = subprocess.run(
            ['docker', 'run', '--rm',
             '--network', 'mobilenetv2-ssd_default',
             '-v', f'{host_dir}/configs:/app/configs',
             '-v', f'{host_dir}/videos:/app/videos',
             '-v', f'{host_dir}/datasets:/app/datasets',
             'mobilenetv2-ssd-etl:latest',
             '--config', '/app/configs/etl/default.yaml',
             '--videos', *video_paths],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ETL container failed:\n{result.stderr}")
        return {'videos_processed': len(video_paths)}

    with open(config_path) as f:
        config = yaml.safe_load(f)
    dashboard_url = config['etl']['ray']['dashboard_url']
    videos_arg = ' '.join(video_paths)
    resp = requests.post(f'{dashboard_url}/api/jobs/', json={
        'entrypoint': f'python -m src.cli.etl --config {config_path} --videos {videos_arg}'
    })
    resp.raise_for_status()
    return resp.json()


def _wait_for_ray(**context):
    if LOCAL_MODE:
        return {'skipped': True}
    ec2_client = boto3.client('ec2', region_name='us-east-1')
    instances = ec2_client.describe_instances(Filters=[
        {'Name': 'tag:Name', 'Values': ['etl-ray-worker']},
        {'Name': 'instance-state-name', 'Values': ['running']}
    ])
    ip = instances['Reservations'][0]['Instances'][0]['PublicIpAddress']
    for _ in range(40):
        try:
            requests.get(f'http://{ip}:8265', timeout=5)
            return ip
        except Exception:
            time.sleep(15)
    raise TimeoutError("Ray did not start in time")


def _send_summary_email(**context):
    if not OWNER_EMAIL:
        return
    ds = context['ds']
    with open(context['params']['config_path']) as f:
        config = yaml.safe_load(f)
    db_url = config['etl']['database']['url']
    parsed = urlparse(db_url)
    conn = psycopg2.connect(
        host=parsed.hostname, port=parsed.port,
        dbname=parsed.path.lstrip('/'),
        user=parsed.username, password=parsed.password
    )
    with conn.cursor() as cur:
        cur.execute("""
            SELECT v.filename, v.duration, v.fps, v.width, v.height,
                   COUNT(DISTINCT f.id) AS frames,
                   COUNT(a.id) AS annotations
            FROM videos v
            LEFT JOIN frames f ON f.video_id = v.id
            LEFT JOIN annotations a ON a.frame_id = f.id
            WHERE DATE(v.created_at) = %s
            GROUP BY v.id
            ORDER BY v.created_at
        """, (ds,))
        rows = cur.fetchall()
        cur.execute("""
            SELECT a.class_name, COUNT(*) AS total
            FROM annotations a
            JOIN frames f ON f.id = a.frame_id
            JOIN videos v ON v.id = f.video_id
            WHERE DATE(v.created_at) = %s
            GROUP BY a.class_name ORDER BY total DESC
        """, (ds,))
        class_rows = cur.fetchall()
    conn.close()

    total_videos = len(rows)
    total_frames = sum(r[5] for r in rows)
    total_annotations = sum(r[6] for r in rows)

    video_rows_html = ''.join(
        f'<tr><td>{r[0]}</td><td>{r[1]:.1f}s</td><td>{r[2]:.1f}</td>'
        f'<td>{r[3]}x{r[4]}</td><td>{r[5]}</td><td>{r[6]}</td></tr>'
        for r in rows
    )
    class_rows_html = ''.join(
        f'<tr><td>{r[0]}</td><td>{r[1]}</td></tr>' for r in class_rows
    )

    html = f"""
    <h2>ETL Pipeline Summary &mdash; {ds}</h2>
    <p><strong>Videos:</strong> {total_videos} &nbsp;
       <strong>Frames:</strong> {total_frames} &nbsp;
       <strong>Annotations:</strong> {total_annotations}</p>
    <h3>Per-Video Breakdown</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
      <thead style="background:#f0f0f0">
        <tr><th>File</th><th>Duration</th><th>FPS</th><th>Resolution</th>
            <th>Frames Sampled</th><th>Annotations</th></tr>
      </thead>
      <tbody>{video_rows_html}</tbody>
    </table>
    <h3>Class Distribution</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
      <thead style="background:#f0f0f0"><tr><th>Class</th><th>Count</th></tr></thead>
      <tbody>{class_rows_html}</tbody>
    </table>
    """
    send_email(to=OWNER_EMAIL, subject=f'ETL Pipeline Complete — {ds}', html_content=html)


with DAG(
    dag_id='etl_pipeline',
    default_args=default_args,
    schedule_interval='0 2 * * *',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    params={'config_path': 'configs/etl/default.yaml'}
) as dag:

    provision_ec2 = BashOperator(
        task_id='provision_ec2',
        bash_command='echo "local mode - skipping EC2 provisioning"' if LOCAL_MODE
            else 'cd /app/infrastructure && terraform apply -auto-approve -target=aws_instance.etl',
    )

    wait_for_ray = PythonOperator(
        task_id='wait_for_ray',
        python_callable=_wait_for_ray,
    )

    run_etl_job = PythonOperator(
        task_id='run_etl_job',
        python_callable=_run_etl_job,
    )

    teardown_ec2 = BashOperator(
        task_id='teardown_ec2',
        bash_command='echo "local mode - skipping EC2 teardown"' if LOCAL_MODE
            else 'cd /app/infrastructure && terraform destroy -auto-approve -target=aws_instance.etl',
        trigger_rule='all_done',
    )

    send_summary = PythonOperator(
        task_id='email_summary',
        python_callable=_send_summary_email,
        trigger_rule='all_done',
    )

    provision_ec2 >> wait_for_ray >> run_etl_job >> teardown_ec2 >> send_summary
