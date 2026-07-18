# Training

`src/training/{engine,optimizer,schedule,amp,ema,checkpoints,resume,shutdown,metrics}.py`

## Engine (`engine.py`) — the main loop

`fit(config, model, priors_cxcywh, train_dataset, validation_dataset, optimizer, ...)`
is the top-level loop CLI's `train.py` calls. Per epoch:

```
train_one_epoch(...)          # one pass over train_dataset, updates weights + EMA
  └─ if epoch % eval_every == 0 or last epoch:
       evaluate(...)          # runs eval, computes mAP, logs health metrics
  └─ checkpoint_manager.save_best(...)   if new best primary metric
  └─ checkpoint_manager.save_last(...)   always, end of epoch
  └─ (if S3 sync configured) upload artifacts + write DynamoDB ledger checkpoint URI
```

`GracefulShutdownException` (raised by `ShutdownHandler` — SIGINT/SIGTERM) is caught at
the `fit()` level: an emergency `save_last` + S3 upload happens before re-raising, so a
killed/preempted training run (e.g. spot instance reclaim) doesn't lose the last epoch.

### `training_step` — one batch, forward + loss

```
building_training_targets(...)   # match GT to priors, encode offsets  [see model-ssd.md]
model(image, training=True)      → predicted_offsets, predicted_logits
build_conf_loss(...)             # per-anchor classification loss, NaN on ignored anchors
select_hard_negatives(...)       # keep only the hardest negatives (neg_pos_ratio)
calculate_final_loss(...)        # multibox_loss: weighted cls + loc loss
```

Every shape transition is guarded by `tf.debugging.assert_equal` — this is deliberate
verbosity given how easy it is for anchor-count mismatches between the head output and
the priors to silently produce wrong-but-runnable code.

### `train_one_epoch` — gradient step details

- `backbone_grad_scale` (default `0.1`): gradients flowing into `model.backbone`
  variables are scaled down before `optimizer.apply_gradients` — a differential
  learning-rate trick so the ImageNet-pretrained backbone moves slower than the
  randomly-initialized SSD head, since it's already in a good place. Implemented as a
  post-hoc gradient multiply (`backbone_mask` computed once via `id()` matching against
  `model.backbone.trainable_variables`) rather than per-parameter-group optimizer LRs.
- The actual step is wrapped in `@tf.function(reduce_retracing=True)` — retracing per
  distinct batch shape would be prohibitively slow; `reduce_retracing` lets TF collapse
  functionally-similar signatures.
- `optimizer.scale_loss(total_loss)` only kicks in if `optimizer` is a
  `LossScaleOptimizer` (i.e. AMP is enabled — see below); otherwise the raw loss is used.
- EMA update (`ema.update(global_step)`) happens every step regardless of `log_every` —
  logging cadence and EMA update cadence are independent.

### `evaluate` / `evaluate_step`

Runs eval under `ema.eval_context(model)` — if EMA is configured to be used at eval time
(`eval_use_ema`), the EMA weights are swapped in for the duration of the eval loop and
restored afterward (see `EMA.eval_context` below). Computes NMS'd detections via
`build_decoded_boxes` (see [model-ssd.md](model-ssd.md#inference-time-post-processing-opspostprocess_tfpy-orchestrationpost_process_orchpy)),
feeds them + GT into `MetricsCollection` for mAP, and periodically logs a large block of
"health" diagnostics (`core/utils.py` — not covered in depth here since they're
debugging aids, not architecture: things like mean background probability, fraction of
zero-detection images, GT/pred box coordinate sanity ranges). These exist because this
model went through real debugging pain around degenerate boxes/collapsed predictions —
if scores or boxes look wrong during training, these are the first metrics to check.

## Optimizer (`optimizer.py`)

`OptimizerFactory.build(config)` — a name-dispatch factory over `sgd` / `adam` / `adamw`,
each pulling its own hyperparameters out of `config["optimizer"]` with defaults. Global
gradient-norm clipping (`grad_clip_norm`) is optional and applied identically across all
three via `global_clipnorm` kwarg.

## LR schedule (`schedule.py`)

Four `tf.keras.optimizers.schedules.LearningRateSchedule` implementations, selected by
`config["scheduler"]["name"]`:

- **`cosine_warmup`** — linear warmup to `base_lr` over `warmup_steps`, then cosine decay
  to `min_lr` over the remaining `total_steps`. The one actually used in practice.
- **`step_decay`** — multiply by `gamma` at each of `milestones` (step-count list).
- **`exponential_decay`** — continuous exponential decay every `decay_interval` steps.
- **`constant`** — flat `base_lr`.

All are step-indexed (not epoch-indexed) — `total_steps`/`warmup_steps` must be given in
optimizer steps, not epochs; the config extraction layer converts epoch-based config
values elsewhere (search for `warmup_epochs` usage in the caller if wiring a new config).

## Mixed precision (`amp.py`)

`AMPContext` wraps `tf.keras.mixed_precision` global-policy setup
(`mixed_float16`/`mixed_bfloat16`/`float32`) and optionally wraps the optimizer in
`LossScaleOptimizer` (dynamic or fixed loss scale). `force_fp32` is a `set[str]` of tags
(passed through as `PrecisionConfig`, see [model-ssd.md](model-ssd.md)) — operations
known to be numerically fragile under fp16 (IoU, box encode/decode, NMS, loss reduction)
can be pinned back to fp32 even while the rest of the model runs at reduced precision.
Note `autocast()` is currently a no-op passthrough context manager — actual dtype
casting happens at the op level via `should_force_fp32`, not through a TF autocast scope.

## EMA (`ema.py`)

Exponential moving average of `model.trainable_variables`, updated every
`update_every` steps after `warmup_steps` have passed. Decay ramps up over the first few
updates (`adjusted_decay = (1+n)/(10+n)`, capped at `decay`) — standard EMA warmup trick
so the average isn't dominated by the (still-garbage) very first weights.

`apply_to(model)` / `restore(model)` swap EMA weights into the live model in place and
back — this mutates `model`'s actual variables temporarily, guarded by an internal
`_backup`/`_is_applied` state machine that raises if you call `apply_to` twice without
an intervening `restore` (prevents silently clobbering the backup). `eval_context(model)`
is the safe way to use this — a context manager that applies EMA on entry and always
restores on exit, even on exception.

## Checkpoints (`checkpoints.py`)

`CheckpointManager` wraps two independent `tf.train.CheckpointManager`s over the same
`tf.train.Checkpoint` bundle (`model`, `optimizer`, `ema`, plus scalar `tf.Variable`s for
epoch/global_step/best_metric/best_epoch):

- **`last/`** — saved every epoch (or every N steps via `save_every_steps`), keeps
  `keep_last_k` most recent (rolling window, for resume-from-crash).
- **`best/`** — saved only when `save_best(metric)` beats the tracked best
  (`mode="max"`/`"min"`), keeps exactly 1 (this is what gets exported to ONNX).

Checkpoint directory naming is fingerprint-based
(`_create_checkpoint_directory_fingerprint`): `runs/<experiment_id>_<fingerprint_short>/
logs/<UTC timestamp>/checkpoints/{last,best}/` — every training invocation of the same
resolved config gets a distinct timestamped log dir, but they all share the fingerprint
prefix so `resume.py`'s scanning can find "the most recent run of this exact config."

## Resume (`resume.py`)

Pure filesystem-scanning helpers, no TF import at module scope for the discovery
functions:
- `discover_checkpoint(dir, target_step=None)` — regex-matches `ckpt-<N>.index` files,
  returns the highest step (or an exact match).
- `find_latest_run_by_fingerprint(runs_root, fingerprint_short)` — scans
  `runs/*/fingerprint.json` for a match, then the newest timestamped log dir under it
  with a real checkpoint. This is how `train.py --resume` finds "continue this exact
  experiment" without the caller needing to know the timestamp.
- `validate_checkpoint_compatibility(saved_config, current_config)` — before restoring,
  diffs `_ARCHITECTURE_KEYS` (`num_classes`, `backbone`, `heads`, `priors`, `input_size`)
  — any difference means variable shapes won't match and restore will corrupt or crash,
  so this is a hard incompatibility. `_TRAINING_KEYS` differences (LR, augmentation,
  scheduler, etc.) are safe-to-differ, just warned about — this is exactly the
  "fine-tune with a lower LR" scenario. AMP enabled/disabled is checked specially since
  it changes the optimizer's own checkpoint variable structure.
- `collect_resumable_runs` / `select_run_interactive` — CLI-facing: lists all
  non-`completed` runs across `runs_root` for a human to pick from (used by an
  interactive `--resume` prompt, not by any automated ledger-driven resume path).

## Shutdown (`shutdown.py`)

`ShutdownHandler` — installs `SIGINT` (and `SIGTERM`, non-Windows only) handlers that set
a `threading.Event` rather than raising immediately, so the training loop can check
`is_requested()` at a safe point (start of each step/epoch) and unwind cleanly (save
checkpoint, upload artifacts) instead of dying mid-`GradientTape`. Must be `register()`ed
from the main thread (raises otherwise) since Python's `signal` module only allows
handlers to be installed there.

## Metrics (`metrics.py`)

`MeanAveragePrecision` — a from-scratch VOC/COCO-style mAP implementation (not
`pycocotools` or a TF/Keras metric):
- **VOC style** (`style="voc"`, the default here) — all-points interpolated AP
  (`_ap_voc`, the standard monotonic-precision-envelope + trapezoid method), one
  IoU threshold (0.5 in this project's config).
- **COCO style** — 101-point interpolation (`_ap_101_point`) averaged across multiple
  IoU thresholds (0.5:0.95), plus separately reported `AP@0.50`/`AP@0.75` — available but
  not what `exp002_cloud_run`'s reported 76.6% mAP used.

`_compute_ap_for_class` is a textbook greedy-matching AP: predictions sorted by score
descending, each claims its best-IoU unmatched GT box if IoU ≥ threshold (else false
positive), cumulative TP/FP → precision/recall curve → AP. `MetricsCollection` wraps
multiple named `MeanAveragePrecision` instances (e.g. different IoU threshold configs)
so `evaluate()` can report several metric suites in one pass — this is also the "metrics
store" the platform's `ARCHITECTURE.md` design references as living in `runs/`.
