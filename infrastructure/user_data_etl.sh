#!/bin/bash
set -euo pipefail

exec > >(tee /var/log/user_data.log) 2>&1
echo "=== User data script started at $(date) ==="

echo ">>> Waiting for apt locks..."
systemctl stop unattended-upgrades || true
while fuser /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/cache/apt/archives/lock /var/lib/dpkg/lock >/dev/null 2>&1; do
    echo "apt lock held, retrying in 5 seconds..."
    sleep 5
done

echo ">>> Verifying NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &> /dev/null; then
    mkdir -p /usr/share/keyrings
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker --set-as-default
    systemctl restart docker
fi

echo ">>> Pulling Docker image: ${docker_image}..."

if [[ "${docker_image}" == *".dkr.ecr."* ]]; then
  REGION=$(echo "${docker_image}" | cut -d. -f4)
  aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$(echo "${docker_image}" | cut -d/ -f1)"
fi

docker pull "${docker_image}"

# --- Starting the Ray head node ---

echo ">>> Starting Ray head node..."
docker run -d \
    --gpus all \
    --name etl-ray \
    --entrypoint ray \
    -p 8265:8265 \
    -p 6379:6379 \
    -e DATABASE_URL="${database_url}" \
    -e AWS_DEFAULT_REGION="${aws_region}" \
    "${docker_image}" \
    start --head --port=6379 --dashboard-host=0.0.0.0 --block

echo ">>> Ray head node started successfully at $(date)."