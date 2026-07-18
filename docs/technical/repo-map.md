# Repo Map & Data Flow

## What this repo is

MobileNetV2 backbone + SSD detection head, trained on VOC (and VisDrone), exported to
ONNX for deployment on a Jetson drone (TensorRT + ROS2 C++ node lives in a separate repo).
This repo is a **submodule of a bigger ml-platform** (see [../ARCHITECTURE.md](../ARCHITECTURE.md)
for the control-plane/data-plane vision) — the pieces here are the "detector" task-type's
actual model, training loop, and export pipeline.

## Directory layout

```
src/
  mobilenetv2ssd/          # the model package (importable, no CLI/IO side effects)
    core/                  # config loading, fingerprinting, precision, logging, exceptions
    models/
      mobilenet_v2/        # backbone: blocks.py, backbone.py (ImageNet weight transplant)
      ssd/
        model.py           # SSD wrapper: backbone + extra pyramid + heads
        fpn.py              # ExtraFeaturePyramid (P6/P7/P8 extra conv levels)
        ops/                # pure functions: anchors, box math, encode, match, loss, postprocess
        orchestration/      # config-driven glue that calls ops/ with values pulled from YAML
      factory.py            # build_ssd_model(config, anchors_per_layer) -> SSD instance
  training/                 # training loop: engine, optimizer, schedule, amp, ema, checkpoints
  datasets/                 # VOC / VisDrone loaders, transforms, collate
  deploy/
    export/                 # export.py (SavedModel) -> convert.py (ONNX) -> validate.py, quantize.py
  etl/                      # video -> frame sampling -> detector consensus -> DB pipeline
  infrastructure/           # DynamoDB experiment ledger, S3 sync (used by the platform, not training itself)
  cli/                      # thin entrypoints: train.py, inference.py, onnx_inference.py, bundle.py, etl.py
tests/
  unit/                     # fast, deterministic — ops, backbone, losses, priors (pytest -m unit)
  integration/               # touches real AWS (DynamoDB) — pytest -m integration
  regression/                 # locks down previously-broken behavior — pytest -m regression
configs/
  deploy/                   # single YAML per deployment target, drives the whole export pipeline
```

See the per-area docs for detail: [model-backbone.md](model-backbone.md),
[model-ssd.md](model-ssd.md), [training.md](training.md), [datasets.md](datasets.md),
[deploy.md](deploy.md), [etl.md](etl.md), [infrastructure.md](infrastructure.md),
[cli.md](cli.md).

## `mobilenetv2ssd/` vs. everything else

`mobilenetv2ssd/` is the one package with **no CLI or filesystem side effects** —
everything is `tf.keras.Model`/pure-function based, driven entirely by a `config: dict`
passed in by the caller. `training/`, `datasets/`, `deploy/`, `etl/` orchestrate it for a
specific job (train, export, run ETL). `cli/*.py` are the thin `argparse` wrappers users
actually invoke.

## Config: the single source of truth

Every stage (`train.py`, `export.py`, `convert.py`, `validate.py`, `inference.py`) is
driven by one resolved config dict, loaded via `src/mobilenetv2ssd/core/config.py::load_config`:

1. An experiment YAML lists `defaults: {component: path/to/base.yaml}` — merged in order.
2. `recipes` can swap in an alternate base file per-component.
3. Top-level keys in the experiment YAML itself override the merged defaults.
4. `overrides` block applies last.
5. CLI `--override key.nested=value` args apply after that (`parse_cli_overrides`).
6. `${VAR}` / `${VAR:-default}` placeholders are substituted from the environment (`inject_env_vars`).
7. Any key named/ending in `root`/`dir`/`path` gets resolved to an absolute path relative
   to `PROJECT_ROOT` (`_resolve_paths`).

The **fingerprint** (`core/fingerprint.py::Fingerprinter`) is a SHA-256 of the
canonicalized resolved config — it's the content-address tying a config to its
DynamoDB ledger row and S3 artifacts (see [../ARCHITECTURE.md §7](../ARCHITECTURE.md)).

**Precision control**: `core/precision_config.py::PrecisionConfig` is a simple
`set[str]` of tags (e.g. `"iou"`, `"box_encode_decode"`, `"nms"`) that must be forced to
fp32 even under mixed precision — passed through most of the `ops/` functions as
`precision_config` and checked via `should_force_fp32(tag, precision_config)`.

## Train → export → inference data flow

```
checkpoint (tf-gpu venv)
  └─ src/deploy/export/export.py  →  saved_model/          (bakes in normalization,
                                                             box decode, softmax)
       └─ src/deploy/export/convert.py (onnx-export venv)  →  model.onnx
            └─ src/deploy/export/validate.py               →  PASS/FAIL parity check
                 └─ src/deploy/export/quantize.py            →  INT8 (PTQ) variant

Inference:
  src/cli/inference.py        — TF SavedModel, image/webcam (tf-gpu venv)
  scripts/onnx_inference.py   — ONNX Runtime, one-off check (onnx-export venv)
```

Input contract end-to-end: **300×300 NHWC float32 [0,1]**, no separate preprocessing
step needed by callers — normalization/decode/softmax are baked into the SavedModel's
serving signature (see [deploy.md](deploy.md)).
