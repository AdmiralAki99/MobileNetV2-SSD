# Dockerfile for MobileNetV2-SSD training

FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "numpy>=1.26.0" \
    "Pillow>=10.0.0" \
    "pyyaml>=6.0" \
    "boto3>=1.34.0"

# Copy source code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Set Python path
ENV PYTHONPATH=/app/src

# Environment variables for configuration
ENV DATA_ROOT=/data/voc
ENV CHECKPOINT_BUCKET=""
ENV CUDA_VISIBLE_DEVICES=0

# Expose TensorBoard port (optional, for debugging single containers)
EXPOSE 6006

# Set working directory
WORKDIR /app/src

# Default entrypoint - run training
ENTRYPOINT ["python", "-m", "cli.train"]

# Default arguments (can be overridden at runtime)
CMD ["--experiment_path", "../configs/experiments/exp002_cloud_run.yaml"]
