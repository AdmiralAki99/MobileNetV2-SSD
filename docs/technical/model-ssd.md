# SSD Head: Architecture, Anchors, Matching, Loss, Post-processing

`src/mobilenetv2ssd/models/ssd/` — `model.py`, `fpn.py`, `ops/*.py`, `orchestration/*.py`

## Layering: ops vs. orchestration

- **`ops/`** — pure, mostly `@tf.function`-decorated math functions. No config
  dicts, no I/O. Every batched op has a matching `_core` (single-example) variant that
  it's built from via flatten/reshape, so ops can be unit-tested at the single-example
  level (see `tests/unit/test_*_tf.py`).
- **`orchestration/`** — one file per pipeline stage. Each pulls its own slice of values
  out of a config `dict` (via a private `_extract_information_from_*_config` helper)
  and calls into `ops/` with concrete arguments. This is the layer `training/engine.py`
  and eval code actually call — it's what makes the ops config-driven without the ops
  themselves knowing about YAML.

## Model graph (`model.py`, `factory.py`)

```
image (B,H,W,3)
  └─ backbone (MobileNetV2)              → {C2, C3, C4, C5}
       feature_maps = [dict[k] for k in config["backbone"]["output_layers"]]  # e.g. [C3, C4, C5]
       base = dict[extra_base]                                                # defaults to last of feature_maps
       └─ ExtraFeaturePyramid(base)       → [P6, P7, P8]   (each a strided Conv2D+ReLU on the previous)
  all_features = feature_maps + extra_features
       ├─ LocalizationHead(all_features)  → pred_offsets  (B, total_anchors, 4)
       └─ ClassificationHead(all_features) → pred_logits   (B, total_anchors, num_classes)
```

`factory.build_ssd_model(config, anchors_per_layer)` reads `config["backbone"]`,
`config["heads"]` and does one dummy forward pass immediately after construction —
this is what forces Keras to build every layer (weight shapes fixed) right away rather
than lazily on first real call, matching the same "build eagerly, Keras 3 doesn't like
save/load on unbuilt models" concern as the backbone (see [model-backbone.md](model-backbone.md)).

`anchors_per_layer` (a `list[int]`, anchor count per feature-map cell, one entry per
level including the extra P6–P8 levels) is **not derived inside `SSD`/`factory`** — it
comes from `anchor_ops_tf.build_priors`/`priors_orch.build_priors_from_config` (below)
and must be computed and passed in consistently with the same config.

### Heads (`ops/heads_tf.py`)

`LocalizationHead` and `ClassificationHead` are structurally parallel: for each feature
map level they optionally apply an initial BatchNorm (level 0 only), an optional
1x1 "squeeze" conv (channel reduction, `squeeze_ratio`), an optional intermediate conv,
then a final prediction conv (`conv3x3` or depthwise-separable, `head_type`) producing
`num_anchors_at_level * 4` (localization) or `num_anchors_at_level * num_classes`
(classification) channels. Output is reshaped per level to `(B, H*W*A, 4 or C)` and
concatenated across levels — this concatenated, anchor-major ordering is the contract
that priors/matching/loss/postprocess all assume (`build_priors` produces priors in the
same per-layer, then-per-cell, then-per-anchor order).

Classification prediction conv uses `RandomNormal(stddev=0.01)` kernel init and
zero bias init (standard SSD/RetinaNet practice — keeps early-training logits near
zero so softmax/sigmoid don't start saturated).

### Extra feature pyramid (`fpn.py`)

`ExtraFeaturePyramid` is deliberately simple — each configured level
(`{name, out_channels, stride, kernel_size}`, default `P6/P7/P8` at stride 2) is a single
`Conv2D(..., activation="relu")` applied to the previous level's output, starting from
`base_feature`. No lateral connections / top-down path — this is closer to SSD's
"extra feature layers" than a true FPN despite the class name.

## Anchors / priors (`ops/anchor_ops_tf.py`, `orchestration/priors_orch.py`)

Priors are precomputed once per `(image_size, strides/feature_map_shapes, scales,
aspect_ratios)` config and reused across the whole training run — they are **not**
learned or recomputed per batch.

```
build_priors(image_size, strides, scales, aspect_ratios, s_min, s_max, include_extra, clip)
  ├─ calculate_feature_map_shapes(image_size, strides)     # if shapes not given explicitly
  ├─ compute_scales_per_layer(scales, n_layers, s_min, s_max, include_extra)
  │     linear interpolation between s_min/s_max (SSD paper formula) unless `scales` given explicitly;
  │     `include_extra=True` adds a second "extra" scale per layer: sqrt(s_k * s_k+1) — the paper's aspect-ratio-1 extra box
  ├─ standardize_aspect_ratios(aspect_ratios, n_layers)     # pads/truncates to n_layers, sorts each level's list to [1.0, >1, <1], dedups
  └─ per layer: build_layer_priors(feature_map_shape, image_size, scales_layer, ratios_layer)
        ├─ make_grid_centers(...)      # cell-center coords, normalized to [0,1]
        ├─ anchors_per_cell(...)       # width/height per (scale, ratio) combo at this layer: w = scale*sqrt(ratio), h = scale/sqrt(ratio)
        └─ tile_layer_anchors(...)     # broadcast anchor shapes across every grid cell → (N, 4) cxcywh
     concatenate_priors(...)           # stack all layers → (total_anchors, 4), clipped to [0,1]
```

`compute_meta` builds a diagnostics dict alongside the priors (anchors-per-layer,
cells-per-layer, a config fingerprint via `core/fingerprint.py`) — mostly used for
sanity-checking anchor counts match head output shapes, not consumed at inference time.

`priors_orch.build_priors_from_config` is the config-driven entry point
`training/engine.py` actually calls: it extracts `model_config["priors"]`, validates it
(`_validate_prior_config` — the biggest function in the file, pure input sanity
checking), computes a fingerprint, and optionally batches the (N,4) priors to
`(B, N, 4)` via `build_priors_batched`. There's a `_cache_priors`/`_get_cached_priors`
pair that is currently a no-op stub (`pass` / `return None`) — priors are recomputed
every call; wiring a real cache (keyed by the fingerprint) is a TODO if this ever shows
up as a bottleneck.

**Format note:** priors are always `cxcywh`, always normalized `[0,1]`. Box ops
(`ops/box_ops_tf.py`) provide `xyxy_to_cxcywh` / `cxcywh_toxyxy` / `to_yxyx` /
`from_yxyx` conversions — TF's `combined_non_max_suppression` wants `yxyx`, everything
else in this repo is `xyxy` or `cxcywh`.

## Matching priors to ground truth (`ops/match_ops_tf.py`, `orchestration/targets_orch.py`)

`match_priors(priors_cxcywh, gt_boxes_xyxy, gt_labels, gt_valid_mask, pos_iou_thresh,
neg_iou_thresh, ...)` per single image:

1. Computes the full `(num_valid_gt, num_priors)` IoU matrix (`iou_matrix_core`).
2. If `center_in_gt=True`, zeroes out IoU for any prior whose center falls outside the
   GT box — an extra filter beyond plain IoU (`_check_for_center_alignment`).
3. `_calculate_matches`: each prior is assigned its highest-IoU GT box.
   `positive_mask` = IoU ≥ `pos_iou_thresh`; `negative_mask` = IoU < `neg_iou_thresh`;
   anything between is `ignore_mask` (excluded from both losses).
4. **Bipartite / low-quality-match rescue** (`allow_low_qual_matches`, default `True`):
   for every GT box, force-assign its single best-matching prior as positive even if
   that IoU is below `pos_iou_thresh` — guarantees every GT box gets at least one
   positive anchor, avoiding "orphaned" ground truths on small/unusual boxes. Conflicts
   (two GT boxes both wanting the same prior) are resolved via `tf.scatter_nd` +
   `argmax`, keeping the higher-IoU claim.
5. Returns matched GT boxes/labels reindexed onto the full `(N,)` prior array (zeros
   where unmatched), plus `pos_mask`/`neg_mask`/`ignore_mask`/`matched_iou`.

`targets_orch.building_training_targets` is the batched, config-driven wrapper: `tf.map_fn`
over the batch dimension calling `match_priors` per image, then
`encode_ops_tf.encode_boxes_batch` to turn matched xyxy boxes into SSD-style regression
targets (`ops/encode_ops_tf.py::encode_boxes_core` — the standard
`tx=(gt_cx-prior_cx)/prior_w/variance_center`, `tw=log(gt_w/prior_w)/variance_size`
parameterization, `variance = (0.1, 0.2)` by default). Padded/absent GT boxes
(all-zero row) get zeroed-out offsets rather than garbage from the log/division.

## Hard negative mining (`ops/match_ops_tf.py::hard_negative_mining`, `orchestration/hard_neg_orch.py`)

Because negatives vastly outnumber positives (most anchors are background), only the
`neg_pos_ratio × num_positives` (clamped by `min_neg`/`max_neg`) *hardest* negatives —
i.e. highest per-anchor classification loss — are kept for the loss. NaN losses are
filtered out before `top_k` (defensive — a hard negative arising from a numerically
broken logit shouldn't be selected). `orchestration/conf_loss_orch.py::build_conf_loss`
computes the raw per-anchor loss (masking out `ignore_mask` anchors as NaN so they can
never be top-k'd) that `hard_neg_orch.select_hard_negatives` then consumes.

## Loss (`ops/loss_ops_tf.py`, `orchestration/loss_orch.py`)

`multibox_loss` = weighted sum of:
- **Classification** — softmax cross-entropy (`softmax_cross_entropy_loss`, a manual
  log-sum-exp implementation, not `tf.nn.softmax_cross_entropy_with_logits`) over
  `positive ∪ (hard-negative-selected) negative` anchors.
- **Localization** — Smooth L1 (or plain L1/L2) over **positive anchors only**, between
  predicted offsets and the encoded targets from `targets_orch`.

Both are normalized by one of `num_pos` (default) / `num_neg` / `num_cls` (pos+neg) /
`num_batch`, then combined via `localization_weight`/`classification_weight`. All the
`_flattened`/`boolean_mask` calls that don't feed into the final loss (visible as
bare expression statements in `multibox_loss`, e.g. an unused
`tf.boolean_mask(predicted_offsets_flattened, negative_mask_flattened)`) are dead code
left over from an earlier version — harmless, but worth knowing they're not doing
anything if you're tracing the loss computation.

## Inference-time post-processing (`ops/postprocess_tf.py`, `orchestration/post_process_orch.py`)

```
decode_and_nms(predicted_offsets, predicted_logits, priors, variances, ...)
  ├─ _decode_boxes           # invert the encode_ops_tf parameterization → xyxy, clipped [0,1]
  ├─ _score_from_logits      # softmax (drop background=index 0) or sigmoid, threshold to 0 below scores_thresh
  ├─ (optional) _filter_small_boxes / _pre_nms_top_k    # cheap pre-filters before NMS
  ├─ convert to yxyx (TF NMS wants yxyx)
  └─ tf.image.combined_non_max_suppression             # per-class NMS, batched
       └─ if softmax: nmsed_classes += 1                # restores original label id (background was sliced out at index 0)
       └─ (optional) _restore_to_image_space             # normalized → pixel coords, if image_meta given
```

**This is the one thing to always remember about NMS here**: `combined_non_max_suppression`
treats every class column independently and equally — background must be stripped
(`scores[..., 1:]`) *before* calling it, and the resulting `class_ids` need `+1` added
back to line back up with the original label space (index 0 = background). This is
already handled inside `decode_and_nms`/`_score_from_logits`, but it's the detail to
check first if class IDs ever look off-by-one somewhere downstream.

`orchestration/post_process_orch.build_decoded_boxes` is the eval-time wrapper: reads
`config["eval"]["nms"]`/`config["eval"]["decode"]`, loads the label map from a classes
text file (`_load_label_map` — softmax mode reserves index 0 for `"background"`; sigmoid
mode starts labels at index 0 since there's no explicit background class), and maps
numeric class IDs to string names for display/logging.
