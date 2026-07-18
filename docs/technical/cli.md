# CLI Entrypoints

`src/cli/{train,inference,onnx_inference,bundle,etl}.py`

All follow the same shape noted in project conventions: `parse_args()` →
`execute_X()` (the real logic) → `if __name__ == "__main__": sys.exit(execute_X())`.
Run with `PYTHONPATH=src` from repo root.

## `train.py` — `execute_training()`

The heaviest CLI file; ties together nearly every other doc in this set. Key structure:

```
execute_training()
  ├─ ShutdownHandler().register()                          [training.md]
  ├─ (if DYNAMODB_TABLE+EXPERIMENT_ID env set) ledger_writer.write_status("running")
  ├─ initialize_framework(args)  → TrainingBundle
  │    ├─ initialize_run_settings   — load_config, compute_fingerprint, build_logger,
  │    │                              build_s3_sync, resolve --resume/--run_from
  │    ├─ create_datasets           — [datasets.md] raw or TFRecord path per config
  │    ├─ create_priors             — build_priors_from_config  [model-ssd.md]
  │    ├─ create_optimizer + create_amp   [training.md]
  │    ├─ build_model               — build_ssd_model  [model-ssd.md]
  │    ├─ create_ema, create_build_checkpoint_manager  [training.md]
  │    └─ (if s3_sync) upload initial run directory (config.json/fingerprint.json etc.)
  ├─ train(framework_opts, shutdown_handler, resume_ckpt_path)
  │    ├─ checkpoint_manager.restore_latest() or restore_from_directory(resume_ckpt_path)
  │    ├─ fit(...)                  [training.md — the actual epoch loop]
  │    ├─ logger.save_model_weights(...)
  │    └─ (if s3_sync) upload_final_artifacts
  └─ finally: write ledger status success/failed, dump status.json, upload run dir, logger.close()
```

### Fingerprinting (`compute_fingerprint`)

Not the same as `core/fingerprint.py::Fingerprinter` used elsewhere in isolation — this
file defines its own `FINGERPRINT_KEYS` (architecture + training-relevant config
sections) and `FINGERPRINT_EXCLUDES` (sub-keys to drop even within an included section —
e.g. `train.diagnostics`, `eval.interval_epochs`/`visualization`, `data.loader`/`root`/
`classes_file`) before hashing. The excludes matter: things like local dataset paths or
logging cadence shouldn't change a run's identity/resumability, only genuinely
architecture/training-affecting values should. `_strip_path_keys` additionally removes
any key `core/config.py::_is_path_key` would have resolved to an absolute path — so the
fingerprint is stable across machines even though `load_config` bakes in
machine-specific absolute paths everywhere else.

### Resume resolution (three independent paths, mutually exclusive via args)

- `--resume` (interactive) — scans `runs_root` via `training/resume.py::collect_resumable_runs`,
  prompts the user to pick one, validates architecture compatibility against the current
  config before allowing it.
- `--run_from <path or s3://...>` — resume from a specific directory or S3 checkpoint
  prefix (downloads via `infrastructure/util.py::download_checkpoint_from_s3` if S3).
- neither flag — `train()` just calls `checkpoint_manager.restore_latest()`, which
  no-ops (fresh start) if nothing's there yet.

### `--print_config` / `--dry_run`

`--print_config` loads and prints the fully-resolved config then exits — the standard
way to sanity-check what a given experiment YAML + overrides actually resolves to before
committing GPU time. `--dry_run` runs the entire `initialize_framework` (dataset/model/
optimizer construction, S3 metadata upload) but exits before the actual `fit()` call —
useful for catching config errors, missing files, or shape mismatches without spending
an epoch's worth of compute.

## `inference.py` — `execute_inference()` (SavedModel / TF path)

Loads a `tf.saved_model` export (see [deploy.md](deploy.md)) and runs either:
- **image mode** (`--image path/or/dir`) — batch of one, annotated output images saved to `--output_dir`.
- **webcam mode** (`--webcam --camera <index or http URL>`) — live OpenCV capture loop with FPS overlay.

Since NMS isn't baked into the SavedModel (see [deploy.md](deploy.md#build_serve_model--what-gets-baked-into-the-savedmodel)),
`run_nms()` here does it client-side: strips background column, converts xyxy→yxyx,
calls the same `ops/postprocess_tf.py::_prepare_nms_inputs`/`_run_batched_nms`/
`_restore_to_image_space` helpers used at training-eval time (see
[model-ssd.md](model-ssd.md)) — so this file is a real consumer of those "internal" ops,
not just a duplicate of training code. `--camera` accepts an IP-camera HTTP stream URL
(per project's WSL2 webcam notes: EpochCam/iPhone virtual driver isn't visible from
WSL2, so an "IP Camera Lite" MJPEG stream is the practical workaround).

## `onnx_inference.py` — same contract, ONNX Runtime path

Mirrors `inference.py` exactly in interface (`--image`/`--webcam`, `--camera`,
`--output_dir`) but loads `model.onnx`/`model_int8.onnx` via
`onnxruntime.InferenceSession` instead of a SavedModel, and implements NMS as **plain
NumPy** (`run_nms` here is a from-scratch per-class greedy NMS loop — deliberately not
importing the TF ops, since this path is meant to validate the ONNX artifact
independent of any TensorFlow machinery, closer to what the eventual C++/TensorRT
consumer would actually do). `--model {fp32,int8}` selects which exported ONNX variant
to run — this is the practical way to eyeball INT8 quantization's accuracy impact
without a full mAP eval pass.

## `bundle.py` — `TrainingBundle` dataclass

Not a CLI itself — a plain data container `initialize_framework` builds and `train()`
consumes, holding every object constructed during setup (`logger`, `fingerprint`,
`config`, `model`, `priors_cxcywh`, both datasets, `optimizer`, `precision_config`,
`ema`, `amp`, `metrics_manager`, `checkpoint_manager`, `s3_client`, plus mutable
`start_epoch`/`global_step`/`best_metric` state). Exists so `train.py`'s many setup
functions don't need to return and thread through a dozen positional values by hand.
Commented-out `InferenceBundle`/`DeploymentBundle` stubs suggest the same pattern was
planned to extend to those paths but hasn't been built yet.

## `etl.py` — `execute_etl()`

Thin wrapper: loads a YAML config, `${VAR}`/`${VAR:-default}` env-expands every string
value in it (`_expand_env_vars` — note this is a **separate, simpler** implementation
than `core/config.py::inject_env_vars`, not shared code — both do effectively the same
substitution but were written independently), then calls `etl/runner.py::run_etl(config["etl"],
video_paths, config_path)`. See [etl.md](etl.md) for what happens after.
