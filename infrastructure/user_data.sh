#!/bin/bash

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ---- Logging ----
# Redirect all output to a log file so you can debug via SSH if something fails
exec > >(tee /var/log/user_data.log) 2>&1
echo "=== User data script started at $(date) ==="

# ---- Install NVIDIA Container Toolkit ----

echo ">>> Waiting for apt locks (unattended-upgrades runs on first boot)..."
systemctl stop unattended-upgrades || true
while fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/cache/apt/archives/lock /var/lib/dpkg/lock >/dev/null 2>&1; do
  echo "apt lock held, retrying in 5s..."
  sleep 5
done
echo ">>> apt locks released"

echo ">>> Installing NVIDIA Container Toolkit..."

# Remove existing keyring if present (idempotent)
rm -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime by default
nvidia-ctk runtime configure --runtime=docker --set-as-default
systemctl restart docker

echo ">>> NVIDIA Container Toolkit installed"

# ---- Pull the Docker image ----

echo ">>> Pulling Docker image: ${docker_image}..."

if [[ "${docker_image}" == *".dkr.ecr."* ]]; then
  # ECR image - need to authenticate
  REGION=$(echo "${docker_image}" | cut -d. -f4)
  aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$(echo "${docker_image}" | cut -d/ -f1)"
fi

docker pull "${docker_image}"
echo ">>> Image pulled successfully"

# ---- Download dataset from S3 ----
# Copy dataset to local SSD for fast I/O during training
echo ">>> Downloading dataset from S3..."
DATA_DIR="/data/${dataset_name}"
mkdir -p "$DATA_DIR"

if [[ "${use_tfrecords}" == "true" ]]; then

  aws s3 sync "s3://${s3_dataset_bucket}/${dataset_name}/" "$DATA_DIR/" --exclude "*" --include "*/shards/*"
  echo ">>> TFRecord shards downloaded to $DATA_DIR"
else
  aws s3 sync "s3://${s3_dataset_bucket}/${dataset_name}" "$DATA_DIR"
  echo ">>> Raw dataset downloaded to $DATA_DIR"
fi

# ---- Create output directories ----
mkdir -p /output/runs
mkdir -p /output/logs

# ---- Pull config library + experiment config from S3 ----

echo ">>> Syncing config library + experiment config from S3..."
mkdir -p /output/configs/experiments
aws s3 sync "s3://${experiment_bucket}/config-library/" /output/configs/
aws s3 cp "${experiment_config}" /output/configs/experiments/experiment.yaml
echo ">>> Config synced"

# ---- Run training ----
echo ">>> Starting training at $(date)..."
docker run \
  --gpus all \
  --rm \
  --name ml-training \
  -e CHECKPOINT_BUCKET="s3://${s3_checkpoint_bucket}" \
  -e DATA_ROOT="/data" \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -e GIT_COMMIT="${git_commit}" \
  -v /data:/data:ro \
  -v /output/runs:/app/src/runs \
  -v /output/logs:/app/src/logs \
  -v /output/configs:/app/configs:ro \
  -p 6006:6006 \
  -e DYNAMODB_EXPERIMENT_TABLE="${dynamodb_table_name}" \
  -e AWS_DEFAULT_REGION="${aws_region}" \
  "${docker_image}" \
  --experiment_path /app/configs/experiments/experiment.yaml \
  --config_root /app/configs \

echo "=== Training completed at $(date) ==="
