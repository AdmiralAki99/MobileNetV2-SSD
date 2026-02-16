# MobileNetV2-SSD

Edge-oriented object detection pipeline built from first principles. Combines a MobileNetV2 backbone with an SSD (Single Shot MultiBox Detector) head for real-time inference on embedded hardware such as NVIDIA Jetson and Hailo.

Built with TensorFlow 2.17, trained on PASCAL VOC, and designed for reproducible, configuration-driven experimentation.

## Table of Contents

- [MobileNetV2-SSD](#mobilenetv2-ssd)
  - [Table of Contents](#table-of-contents)
  - [Architecture Overview](#architecture-overview)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Install](#install)
    - [Quick Training Run](#quick-training-run)
  - [Configuration System](#configuration-system)
    - [Example experiment config](#example-experiment-config)
    - [Fingerprinting](#fingerprinting)
  - [Training](#training)
    - [Pipeline](#pipeline)
    - [Key features](#key-features)
    - [Training output](#training-output)
  - [Infrastructure](#infrastructure)
    - [Docker](#docker)
    - [Parallel experiments with Docker Compose](#parallel-experiments-with-docker-compose)
    - [S3 integration](#s3-integration)
    - [EC2 spot training with Terraform](#ec2-spot-training-with-terraform)
  - [Testing](#testing)
  - [Notebook-Driven Development](#notebook-driven-development)
  - [Deployment](#deployment)
  - [Project Status](#project-status)

---

## Architecture Overview

```
Input Image [B, 300, 300, 3]
        │
        ▼
┌──────────────────┐
│  MobileNetV2     │   Inverted residual blocks with depthwise separable convolutions.
│  Backbone        │   Width multiplier (alpha) for model scaling.
│                  │   Outputs multi-scale features: C3, C4, C5
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Extra Feature   │   Stride-2 convolutions generating P6, P7, P8
│  Pyramid         │   for detecting objects at additional scales.
└──────┬───────────┘
       │
       ├──────────────────────┐
       ▼                      ▼
┌──────────────┐    ┌──────────────────┐
│ Localization │    │ Classification   │
│ Head         │    │ Head             │
│              │    │                  │
│ [B, N, 4]    │    │ [B, N, 21]       │
│ box offsets  │    │ class logits     │
└──────────────┘    └──────────────────┘
```

Six feature maps at different resolutions feed into shared-weight prediction heads. Prior (anchor) boxes are generated per feature map cell, and the network predicts offsets and class scores for each prior.

**Training pipeline:** Target assignment via IoU-based matching, hard negative mining (3:1 ratio), and MultiBox loss (smooth L1 + cross-entropy). Supports AMP (mixed precision), EMA (exponential moving average), and cosine-annealed learning rate with warmup.

---

## Project Structure

```
├── configs/                    # Hierarchical YAML configuration
│   ├── base/                   #   Reusable component defaults
│   │   ├── augmentations/      #     Data augmentation presets
│   │   ├── backbones/          #     MobileNetV2 config
│   │   ├── checkpoint/         #     Checkpoint retention policy
│   │   ├── heads/              #     SSD head architecture
│   │   ├── losses/             #     Loss function selection
│   │   ├── optimizers/         #     AdamW / SGD + schedulers
│   │   ├── priors/             #     Anchor box grid settings
│   │   └── ...
│   ├── data/                   #   Dataset configs (VOC 224 / 300)
│   ├── deploy/                 #   Edge deployment (Jetson TensorRT)
│   ├── engine/                 #   Training engine settings
│   ├── experiments/            #   Full experiment definitions
│   ├── model/                  #   End-to-end model configs
│   └── train/                  #   Training workflow configs
│
├── src/
│   ├── cli/                    # Entry points
│   │   └── train.py            #   Main training CLI
│   ├── datasets/               # Data loading and transforms
│   │   ├── voc.py              #   PASCAL VOC 2012 parser
│   │   ├── transforms.py       #   Augmentations (photometric, crop, flip)
│   │   └── collate.py          #   tf.data pipeline creation
│   ├── infrastructure/         # Cloud utilities
│   │   └── s3_sync.py          #   S3 checkpoint upload / download
│   ├── mobilenetv2ssd/
│   │   ├── core/               # Shared utilities
│   │   │   ├── config.py       #     Hierarchical config loader
│   │   │   ├── fingerprint.py  #     Reproducibility hash
│   │   │   ├── logger.py       #     Structured logging + TensorBoard
│   │   │   └── precision.py    #     FP32 enforcement for sensitive ops
│   │   └── models/
│   │       ├── mobilenet_v2/   #     Backbone (inverted residuals)
│   │       ├── ssd/            #     Heads, priors, matching, losses, NMS
│   │       └── factory.py      #     Model builder
│   └── training/               # Training loop
│       ├── engine.py           #   fit(), train_one_epoch(), training_step()
│       ├── optimizer.py        #   Optimizer factory
│       ├── schedule.py         #   LR warmup + cosine annealing
│       ├── amp.py              #   Mixed precision context
│       ├── ema.py              #   Exponential moving average
│       ├── checkpoints.py      #   Save / restore state
│       ├── resume.py           #   Resume from local or S3 checkpoint
│       └── metrics.py          #   VOC mAP @ 0.5
│
├── infrastructure/             # Cloud deployment
│   ├── main.tf                 #   Terraform: EC2 spot + IAM + S3
│   ├── QUICKSTART.md           #   Step-by-step EC2 training guide
│   └── DOCKER_USAGE.md         #   Docker / docker-compose guide
│
├── tests/
│   ├── unit/                   # 12 unit test modules
│   └── integration/            # Multi-component integration tests
│
├── notebooks/                  # Notebook-driven development (see below)
├── Dockerfile                  # TF 2.17-gpu training image
├── Dockerfile.tensorboard      # TensorBoard S3-sync image
├── docker-compose.yml          # Parallel experiments + monitoring
├── Makefile                    # dev, test, lint, format shortcuts
└── pyproject.toml              # Project metadata and dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- TensorFlow 2.17.0 (GPU recommended)
### Install

```bash
# Production dependencies
pip install -e .

# With dev tools (pytest, coverage)
pip install -e ".[dev]"

# With cloud support (boto3 for S3)
pip install -e ".[cloud]"
```

### Quick Training Run

```bash
python -m cli.train \
  --experiment_path configs/experiments/exp001_baseline.yaml \
  --config_root configs/
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--resume` | Resume from the latest checkpoint in the run directory |
| `--run_from <path>` | Resume from a specific checkpoint (local path or `s3://...`) |
| `--print_config` | Print the fully merged config and exit |
| `--dry_run` | Initialize everything (model, data, optimizer) without training |

---

## Configuration System

Configs are hierarchical YAML files merged at runtime. An experiment config references base component configs and can override any value.

```
configs/experiments/exp001_baseline.yaml
        │
        │  defaults:
        │    backbone: base/backbones/mobilenetv2.yaml
        │    train:    base/train/default.yaml
        │    losses:   base/losses/ssd_loss.yaml
        │    ...
        │
        ▼
   Merged Config  ◄── CLI overrides (key.path=value)
                  ◄── Environment variables (${VAR:-default})
```

### Example experiment config

```yaml
experiment:
  id: exp001
  name: mobilenetv2_ssd_baseline
  tags: [baseline, mobilenetv2, voc]

defaults:
  backbone: base/backbones/mobilenetv2.yaml
  train: base/train/default.yaml
  optimizer: base/optimizers/adamw_cosine.yaml
  losses: base/losses/ssd_loss.yaml

overrides:
  train:
    epochs: 50
    batch_size: 3
```

### Fingerprinting

Each run is fingerprinted by hashing the architecture-defining config keys (backbone, heads, priors, num classes, input size). This produces a deterministic run directory name like `exp001_a1b2c3d4` and enables automatic checkpoint compatibility validation when resuming.

---

## Training

### Pipeline

1. **Config merge** — experiment YAML + base defaults + CLI overrides
2. **Fingerprint** — hash architecture params for the run directory
3. **Dataset creation** — VOC parser, transforms, `tf.data` pipeline with padding and prefetch
4. **Prior generation** — anchor boxes at 6 scales with configurable aspect ratios
5. **Model build** — backbone feature extraction, extra pyramid levels, prediction heads
6. **Optimizer + scheduler** — AdamW (or SGD) with linear warmup + cosine annealing
7. **Training loop** — `tf.GradientTape`, AMP autocast, EMA updates, per-epoch checkpointing
8. **Evaluation** — VOC mAP @ IoU 0.5, best-metric checkpointing
9. **S3 sync** — upload checkpoints, logs, and metrics after each epoch

### Key features

- **Mixed precision (AMP):** `mixed_float16` policy with dynamic loss scaling. Sensitive operations (loss reduction, NMS, IoU) are forced to FP32.
- **EMA:** Exponential moving average of model weights with configurable decay and warmup period. EMA weights can be used for evaluation.
- **Hard negative mining:** Selects the hardest negative priors at a configurable ratio (default 3:1) to balance the classification loss.
- **Graceful shutdown:** Catches SIGTERM, saves a checkpoint, uploads to S3, and writes a `status.json` before exiting.

### Training output

```
runs/
└── exp001_a1b2c3d4/
    ├── config.json           # Full merged config snapshot
    ├── fingerprint.json      # Architecture hash
    ├── status.json           # success | failed
    ├── args.json             # CLI arguments
    └── logs/
        └── <timestamp>/
            ├── training.log
            ├── metric_history.json
            └── events.out.tfevents.*   # TensorBoard
```

---

## Infrastructure

### Docker

Two container images are provided:

| Image | Base | Purpose |
|-------|------|---------|
| `Dockerfile` | `tensorflow/tensorflow:2.17.0-gpu` | Training with GPU support |
| `Dockerfile.tensorboard` | — | TensorBoard syncing logs from S3 |

### Parallel experiments with Docker Compose

```
┌──────────────────┐
│   TensorBoard    │ ◄── syncs from S3 every 60s
│   localhost:6006 │
└────────┬─────────┘
         │
    ┌────┴────┐
    │   S3    │
    └────┬────┘
         │ uploads after each epoch
    ┌────┴────┬──────────┐
    │         │          │
┌───┴───┐ ┌──┴────┐  ┌───┴───┐
│exp001 │ │exp002 │  │exp003 │   ← one GPU each
│GPU 0  │ │GPU 1  │  │GPU 2  │
└───────┘ └───────┘  └───────┘
```

```bash
# Set environment
cp .env.example .env    # fill in AWS creds + dataset path

# Launch everything
docker-compose up -d

# Watch a specific experiment
docker-compose logs -f training-exp001

# Monitor in browser
# http://localhost:6006

# Tear down
docker-compose down
```

Add more experiments by duplicating a service block in `docker-compose.yml` with a different GPU ID and experiment config.

### S3 integration

The training loop automatically syncs to S3 when credentials are configured:

- **Upload:** checkpoints, logs, and metrics after each epoch
- **Download:** restore checkpoints for resuming (`--run_from s3://bucket/path`)
- **TensorBoard:** the TensorBoard container polls S3 and serves logs locally

### EC2 spot training with Terraform

The `infrastructure/` directory contains Terraform configs for launching GPU spot instances:

```bash
cd infrastructure/
terraform init
terraform plan     # preview (no cost)
terraform apply    # launches g5.xlarge (~$0.16/hr spot)

# When done:
terraform destroy  # stops billing, keeps S3 data
```

The instance bootstraps automatically: installs NVIDIA toolkit, pulls the Docker image, downloads the dataset from S3, and starts training. See [infrastructure/QUICKSTART.md](infrastructure/QUICKSTART.md) for the full walkthrough.

---

## Testing

```bash
make test               # unit tests (default)
make test-integration   # integration tests
make test-all           # everything
make test-cov           # unit tests with coverage
```

Unit tests cover all core components:

| Module | What it tests |
|--------|--------------|
| `test_backbone_tf` | MobileNetV2 output shapes and feature extraction |
| `test_heads_tf` | Classification / localization head outputs |
| `test_priors_ops_tf` | Anchor grid generation |
| `test_match_ops_tf` | Prior-to-ground-truth IoU matching |
| `test_encode_ops_tf` | Box coordinate encoding / decoding |
| `test_box_ops_tf` | Box format conversions |
| `test_loss_ops_tf` | Loss function values |
| `test_postprocess_ops_tf` | NMS and detection decoding |
| `test_metrics_tf` | mAP computation |
| `test_amp_tf` | Mixed precision context and loss scaling |

---

## Notebook-Driven Development

Every component was implemented and validated in a Jupyter notebook before being promoted to `src/`. This ensures correctness through visualization and incremental testing.

**Core primitives (01-08):** Backbone verification, bounding box ops, encoding logic, SSD heads.

**Orchestration layer (09-13):** Prior/anchor grid visualization, ground-truth matching, hard negative mining ratios, MultiBox loss convergence.

**System integration (14-25):** Model factory, post-processing (NMS + decoding), metrics manager, LR scheduler, checkpoint manager, training step, full SSD forward pass.

---

## Deployment

Edge deployment configs target NVIDIA Jetson with TensorRT:

```yaml
# configs/deploy/mobilenetv2_ssd_voc_jetson.yaml
deploy:
  input:
    size: [300, 300, 3]
    format: NCHW
  post_processing:
    score_threshold: 0.3
    nms_iou_threshold: 0.5
    max_detections: 3
  runtime:
    precision: FP16
    batch_size: 1
```

The deployment workflow: train on GPU, export to SavedModel, convert to TensorRT engine, deploy with the preprocessing/postprocessing config above.

---

## Project Status

This project is under active development. See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for a detailed breakdown of completed, in-progress, and planned work.

**Completed:** Core SSD architecture, training pipeline with AMP/EMA, checkpoint management with S3 resume, Docker + Terraform infrastructure, configuration system, VOC mAP evaluation.

**In progress:** Evaluation CLI, deployment export tooling.

**Planned:** COCO mAP metrics, quantization-aware training, multi-scale training, ROS2 runtime integration.
