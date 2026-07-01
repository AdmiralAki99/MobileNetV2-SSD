import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Environment paths
TF_PYTHON = os.getenv("TF_PYTHON", "")
ONNX_PYTHON = os.getenv("ONNX_PYTHON", "")

# Creating the config macros to use in the api and dashboard easily

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "ml-experiment-ledger")
CHECKPOINT_BUCKET = os.getenv("S3_BUCKET", "akhilesh-ml-checkpoints")
ARTIFACTS_BUCKET = os.getenv("ARTIFACTS_BUCKET", "akhilesh-ml-artifacts")
DATASET_BUCKET = os.getenv("DATASET_BUCKET", "akhilesh-ml-datasets")
IAM_INSTANCE_PROFILE = os.getenv("IAM_INSTANCE_PROFILE", "ml-training-profile")
KEY_PAIR_NAME = os.getenv("KEY_PAIR_NAME", "")
DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "mobilenetv2-ssd:latest")
ETL_DB_URL = os.getenv("ETL_DATABASE_URL", "postgresql://airflow:airflow@localhost:5432/etl_db")
AIRFLOW_DB_URL = os.getenv("AIRFLOW_DB_URL", "postgresql://user:password@postgres:5432/etl_db")
EXPERIMENT_BUCKET = os.getenv("EXPERIMENT_BUCKET", "akhilesh-ml-experiments")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Creating macros for all the important directories for easier indexing

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
INFRA_DIR = PROJECT_ROOT / "infrastructure"
SRC_DIR = PROJECT_ROOT / "src"
