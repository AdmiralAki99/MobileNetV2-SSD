# Datasets

`src/datasets/{base,voc,vis_drone,transforms,collate}.py`

## Contract: `DetectionSample` (`base.py`)

Every dataset loader produces the same dataclass regardless of source format:

```python
DetectionSample(image: np.ndarray[H,W,3] float32,
                 boxes: np.ndarray[N,4] float32,   # xyxy, PIXEL coords (not yet normalized)
                 labels: np.ndarray[N] int32,       # >= 1  (0 is reserved for background)
                 image_id: str, path: str, orig_size: (H, W))
```

`BaseDetectionDataset.__getitem__` always runs `_load_sample` → `_clean_boxes` (clips to
image bounds, drops degenerate/NaN boxes) → optional `.validate()` (strict shape/dtype/
label-range assertions — disable with `validate=False` for speed once a dataset is known-good).

`create_dataset_from_config(config, split)` is the factory: dispatches on
`config["data"]["dataset_name"]` (`"voc"` / `"vis_drone"`) to the concrete loader below.
`iter_annotations()` (yields `(boxes, (H,W))` per image, no image decode) exists
specifically for anchor/prior K-means clustering over box dimensions — it's cheap because
it skips loading pixels.

## `VOCDataset` (`voc.py`)

Standard Pascal VOC layout: `root/JPEGImages/*.jpg`, `root/Annotations/*.xml`,
`root/ImageSets/Main/{split}.txt` (image-id list, one per line). Parses each XML via
`xml.etree.ElementTree`, skips `difficult` objects unless `use_difficult=True`, skips
unknown class names and malformed/degenerate (`x2<=x1` or `y2<=y1`) boxes at parse time
(belt-and-suspenders with `_clean_boxes`).

## `VisDroneDataset` (`vis_drone.py`)

Layout: `root/VisDrone2019-DET-{split}/{images,annotations}/`. Annotation format is CSV
per line: `bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncated,occlusion`
— converted from `(left,top,w,h)` to `xyxy` here. Filters `category==0` (ignored region),
`score==0` (per the VisDrone spec, ignored annotation), and `category==11` (VisDrone's
"others" catch-all class) — these three are dataset-specific noise, not detectable objects.

## Transform pipeline (`transforms.py`)

Two config-driven pipelines are assembled by `build_train_transforms(config)` /
`build_validation_transforms(config)`, each a `Compose` of small stateless-ish callables
operating on `(image, target_dict)`:

**Standardize stage** (`to_float32` → `sanitize_boxes` → `scale_01`) — always first,
converts to model-ready dtypes/range: `ToFloat32`, `ClipAndFilterBoxes` (drops
now-too-small boxes below `min_size` pixels, re-filtering labels in lockstep), `Scale01`
(divide by 255 → `[0,1]`).

**Augmentation stage** (train only, each independently toggleable via
`{name}.enabled` + `.prob`) — `PhotometricDistort` (brightness/contrast/saturation/hue
jitter, randomized order via `tf.cond` branching so it stays graph-mode compatible) and
`RandomHorizontalFlip` (flips both image and box x-coordinates together). Config
scaffolding exists for `random_iou_crop`/`random_expand` (classic SSD augmentations) but
both are **unimplemented no-ops** (`case "random_iou_crop": pass`) — enabling them in
config currently does nothing silently. Worth implementing if training needs the extra
scale-invariance they'd normally provide.

**Final stage** (`resize` → `normalize`) — `Resize` supports `"stretch"` (independent
x/y scale, what this project's 300×300 config actually uses) or `"letterbox"` (uniform
scale + padding, preserves aspect ratio) modes; `Normalize` applies ImageNet mean/std
(`[0.485,0.456,0.406]`/`[0.229,0.224,0.225]` defaults) — note this happens **inside the
dataset pipeline** at train/eval time, whereas at inference/export time normalization is
instead baked into the SavedModel serve wrapper (see [deploy.md](deploy.md)) — same
constants, two different places they get applied depending on which path you're in.

Optional `NormalizeBoundingBoxes` (→ `[0,1]` relative to image size) runs last if
`output_box_norm=True` — this is what feeds into prior matching, which expects
normalized `xyxy` (see [model-ssd.md](model-ssd.md)).

## `tf.data` pipeline construction (`collate.py`)

Two independent data-loading paths, selected by which function the caller invokes
(both eventually converge on the same padded-batch shape):

1. **In-memory generator path** — `create_training_dataset(dataset, config, transform)` /
   `create_validation_dataset(...)`: wraps a `BaseDetectionDataset` instance via
   `tf.data.Dataset.from_generator(dataset.generator, output_signature=_OUTPUT_SIGNATURE)`
   — every sample is decoded/transformed on the fly, per-epoch, in Python. Simple, but
   the whole dataset is walked through Python generator overhead each epoch.
2. **TFRecord path** — `create_training_dataset_from_tfrecords(config, shard_paths, transform)`:
   reads pre-serialized `.tfrecord` shards (schema: `_TFRECORD_FEATURE_DESCRIPTION` —
   JPEG bytes + box/label var-len features + metadata), parsed via `_parse_tfrecord`.
   Faster (native TF I/O, no Python generator per-sample), but requires shards to already
   exist under `root/shards/{train_split,val_split}/` — this is the format the
   TFRecords+stats "dataset factory" work referenced in project memory targets.

Both paths: optional `.shuffle()` → optional `.repeat()` (train only) →
`.map(apply_transform)` → **`padded_batch`** (variable per-image box counts padded to the
batch max, or `max_boxes` if configured) → `.map(_create_gt_mask)` → optional
`.prefetch(AUTOTUNE)`.

**Padding contract**: images pad with `0.0`, boxes pad with `-1.0` (not `0.0` — an
all-zero box would look like a degenerate real box at the origin; `-1` is
unambiguously not a valid coordinate), labels pad with `0` (the reserved "background"
index). `_create_gt_mask` derives `gt_mask = labels > 0` — this is the boolean mask
consumed everywhere downstream (`targets_orch.building_training_targets`'s
`gt_valid_mask`, `metrics.py`'s ground-truth filtering) to know which entries in a padded
batch are real boxes vs. padding, rather than re-deriving it from the sentinel box value.
