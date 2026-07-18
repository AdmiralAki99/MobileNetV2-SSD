# Deploy: Export → Convert → Validate → Quantize

`src/deploy/{__init__.py,export/{export,convert,validate,quantize}.py}`

Single config drives the whole pipeline: `configs/deploy/mobilenetv2_ssd_voc_jetson.yaml`
loaded via `deploy/__init__.py::load_deploy_config` (plain `yaml.safe_load`, no merging —
unlike training's `core/config.py::load_config`, this one is flat/self-contained). Every
stage below takes `--deploy_config` and an optional `--output_dir` override.

## Stage 1 — `export.py`: checkpoint → SavedModel

```
execute_export()
  ├─ load_deploy_config(deploy_config_path)
  ├─ load_config(deploy_config["experiment_path"])          # the ORIGINAL training config
  ├─ build_priors_from_config(experiment_config)              # same priors as training
  ├─ build_ssd_model(experiment_config, anchors_per_layer)    # same architecture as training
  ├─ build_ema(experiment_config, model)                      # EMA wrapper (may or may not be used)
  ├─ download_checkpoint(args.checkpoint)                     # `aws s3 sync` if checkpoint is an s3:// URI
  ├─ tf.train.Checkpoint(model=model, ema=ema).restore(...)
  ├─ build_serve_model(model, priors, deploy_config)           # wraps model in a tf.function w/ baked-in pre/post-processing
  └─ with ema.eval_context(model):
        tf.saved_model.save(model, signatures={"serving_default": serve})
     np.save(priors_cxcywh.npy)
     (smoke test: reload the SavedModel, run one dummy forward pass, print output shapes)
```

**Why the experiment config is re-loaded here** rather than trusting the deploy config
alone: the model architecture (backbone type, head config, num_classes, priors) must
exactly match what the checkpoint was trained with — `deploy_config["experiment_path"]`
points back at that original training YAML so `build_ssd_model`/`build_priors_from_config`
reconstruct the identical graph the checkpoint's weights fit into. The **deploy config**
only supplies deployment-specific values: preprocessing mean/std, priors variances,
target opset, output paths, num_classes for the runtime.

### `build_serve_model` — what gets baked into the SavedModel

The returned `@tf.function` (the `serving_default` signature) does, in order:
**normalize** (`(x - mean) / std`) → **forward pass** → **decode boxes** (manually
re-implements the offset→xyxy math inline, rather than importing
`ops/postprocess_tf.py::_decode_boxes` — a duplicate implementation to know about if the
decode formula ever changes, since it needs to be updated in two places) → **softmax**
the raw class logits. **NMS is deliberately not included** — the SavedModel/ONNX output
is raw `(boxes, scores)`; NMS is expected to run downstream (the ROS2/TensorRT C++ node,
per project context) since NMS ops don't reliably export/execute the same way across
ONNX Runtime / TensorRT backends. This is why callers of the exported model (`inference.py`,
`scripts/onnx_inference.py`) still need their own NMS step — see [cli.md](cli.md).

Input contract: `(B, H, W, 3)` float32, values in `[0, 1]` — the caller does **not**
pre-normalize; normalization is inside the graph. This is the single most important
invariant for anyone consuming the exported model from a new client.

## Stage 2 — `convert.py`: SavedModel → ONNX

Thin wrapper around `python -m tf2onnx.convert --saved-model ... --output model.onnx
--opset <deploy_config.deploy.runtime.opset>`, run as a `subprocess` rather than calling
`tf2onnx`'s Python API directly — this is why `convert.py` needs the separate
`onnx-export` venv (per project convention) rather than running in the same process as
`export.py`'s `tf-gpu` env: `tf2onnx` and the training TF build don't need to coexist in
one interpreter this way.

## Stage 3 — `validate.py`: ONNX parity/shape check

Loads the ONNX model via `onnxruntime.InferenceSession`, runs one dummy
all-zeros forward pass at the configured batch size, and asserts `boxes.shape ==
(B, num_anchors, 4)` / `scores.shape == (B, num_anchors, num_classes)`. This is a
**shape sanity check**, not a numerical parity check against the TF SavedModel — it
confirms the ONNX graph is loadable and structurally correct, not that its outputs
numerically match the TF model bit-for-bit. Prints `PASS`/`FAIL` and exits 0/1
accordingly — this is the gate `scripts/preflight_check.py`/CI would key off of.

## Stage 4 — `quantize.py`: ONNX → INT8 (post-training static quantization)

Uses `onnxruntime.quantization.quantize_static` with `QuantFormat.QDQ` (quantize-dequantize
node pairs — the format TensorRT's INT8 engine builder expects, vs. the alternative
`QOperator` format) and `QInt8` for both weights and activations, `per_channel=False`.

`_ImageCalibrationReader` implements `CalibrationDataReader`: feeds real images from
`--calibration_images` (default 200) through the *same* preprocessing as the training
dataset loader (resize bilinear → `/255.0` → batch dim) so the calibration statistics
(activation min/max ranges used to pick INT8 scale factors) reflect real input
distributions. Runs the same `validate_onnx` shape check on the quantized output as a
post-quantization sanity gate.

**Known accuracy caveat** (from project history): INT8 PTQ measurably degrades class
discrimination on this model — this is a known, accepted tradeoff for the Jetson
deployment's speed/memory budget, not a bug to chase. If quantized-model accuracy
regresses further than expected, check calibration image count/diversity before
suspecting the quantization code itself.

## End-to-end command sequence

```bash
# tf-gpu venv
python -m src.deploy.export.export --deploy_config configs/deploy/mobilenetv2_ssd_voc_jetson.yaml --checkpoint runs/.../checkpoints/best

# onnx-export venv
python -m src.deploy.export.convert  --deploy_config configs/deploy/mobilenetv2_ssd_voc_jetson.yaml
python -m src.deploy.export.validate --deploy_config configs/deploy/mobilenetv2_ssd_voc_jetson.yaml
python -m src.deploy.export.quantize --deploy_config configs/deploy/mobilenetv2_ssd_voc_jetson.yaml --calibration_images path/to/images
```

Also see `scripts/export_pipeline.sh` / `scripts/export_model.py` /
`scripts/compare_inference.sh` / `scripts/preflight_check.{py,sh}` at repo root for the
scripted/CI-facing wrappers around these same four stages (untracked/local-only per
`.gitignore`, but worth checking if you need the exact invocation flags used last).
