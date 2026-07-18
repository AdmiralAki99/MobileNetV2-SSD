# ETL: Video → Frames → Multi-Model Consensus → TFRecords

`src/etl/{pipeline,frame_sampler,detectors,consensus,writer,db,runner}.py`

Purpose: turn raw drone footage into labeled training data automatically, using an
ensemble of off-the-shelf detectors "voting" on what's actually in each frame — this is
how new VisDrone-style training data gets bootstrapped without hand-labeling. Distinct
from the training/model code — this pipeline **produces** datasets, it doesn't consume
them for training directly (though its `TFRecordWriter` output format matches what
`datasets/collate.py`'s TFRecord path expects).

## Pipeline (`pipeline.py`, `runner.py`)

```
run_etl(config, video_paths, config_path)
  ├─ mode="local": ray.init() + N x ETLWorker.remote() actors, round-robin video assignment
  └─ mode=cloud:   submits `python -m src.cli.etl ...` as a Ray job via JobSubmissionClient

ETLWorker.process_video(video_path)   [runs per Ray actor]
  ├─ FrameSampler.sample(video_path)              → scene-change-filtered frames + video metadata
  ├─ Video row inserted (status="processing")
  ├─ per sampled frame:
  │    ├─ run all 3 detectors → detections_per_model
  │    ├─ ConsensusEngine.compute(...)             → cross-model-agreed annotations only
  │    ├─ (skip frame if no consensus annotations)
  │    ├─ TFRecordWriter.write(...)                 → append to current shard
  │    └─ Frame + Annotation rows inserted
  └─ Video row updated (status="completed")
```

Each `ETLWorker` is a **Ray actor** (`@ray.remote`) — one instance per worker holds its
own loaded detector models, DB session, and TFRecord writer state, so models aren't
reloaded per video. `dataset_name` is randomly generated per worker
(`{adjective}_{noun}_{date}`, e.g. `swift_falcon_20260717`) — this groups a given
worker's entire output run under one dataset name, distinct across workers/runs.

## Frame sampling (`frame_sampler.py`)

Not a fixed frame-rate sample — `FrameSampler.sample` strides through the video every
`stride_frames` (default 30), computes a grayscale histogram per candidate frame, and
keeps it only if its **Bhattacharyya distance** from the previous kept frame's histogram
exceeds `scene_change_threshold` (default 0.35). This is a cheap proxy for "this frame
looks meaningfully different from the last one" — avoids wasting detector compute (and
downstream training data) on near-duplicate consecutive frames from a slow-moving drone
shot. Caps at `max_frames_per_video` (default 100). Supports `s3://` video paths
(downloaded to a temp file first via `boto3`).

## Detectors (`detectors.py`)

Three independent off-the-shelf models, run on every sampled frame, each wrapped to a
common `Detection(box, class_id, class_name, confidence)` output normalized to the
**VisDrone class taxonomy** (not COCO, not this project's own model):

- **`YOLODetector`** (`ultralytics.YOLO`) — COCO-pretrained; only detections whose COCO
  class is in `COCO_TO_VISDRONE` (a hardcoded 6-class subset mapping — e.g. COCO
  `person→pedestrian`, `car→car`, `bus→bus`) are kept; everything else is silently dropped.
- **`RTDETRDetector`** (`ultralytics.RTDETR`) — same COCO→VisDrone mapping/filter, different backbone/architecture.
- **`GroundingDINODetector`** (HuggingFace `AutoModelForZeroShotObjectDetection`,
  `IDEA-Research/grounding-dino-tiny` by default) — a **text-prompted** open-vocabulary
  detector; its free-text output labels are matched against `TEXT_TO_VISDRONE` (case-
  insensitive) rather than a fixed class ID mapping.

Using three architecturally different detectors (CNN-anchor YOLO, transformer-based
RT-DETR, and prompt-based Grounding DINO) is deliberate — the point is that their
individual failure modes are uncorrelated, so agreement between them is a stronger
"this is really an object" signal than any one model's confidence score alone.

## Consensus (`consensus.py`)

`ConsensusEngine.compute(detections_per_model)` — this is the actual "voting" step:

1. Flattens all detections from all models into one list, tracking which model each came from.
2. For every **pair** of models, builds an IoU matrix between their detections and runs
   `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) — the *optimal* one-to-one
   matching between the two models' boxes, not greedy nearest-neighbor.
3. Any matched pair with IoU ≥ `iou_threshold` (default 0.5) gets union-find merged into
   the same cluster (a lightweight disjoint-set, `_find`/`_union` with path compression).
4. Clusters with fewer than `min_votes` (default 2) distinct contributing models are
   dropped — a detection only one model saw is treated as noise, not a real object.
5. Surviving clusters are merged into one `ConsensusAnnotation`: confidence-weighted
   average box, majority-vote class ID (with a tiebreak toward whichever class has more
   supporting detections), mean confidence as the reported `consensus_score`.

This is genuinely a small multi-object-tracking-style data-association problem solved
per-frame (no temporal tracking across frames) — worth remembering if debugging why two
overlapping detections from different models did or didn't merge: check the *pairwise*
IoU threshold and whether `min_votes` allowed a 2-model agreement through.

## Storage (`db.py`, `writer.py`)

**Postgres/SQLAlchemy** (`db.py`) — `Video` → `Frame` → `Annotation` (one-to-many chains,
FK-linked), plus a `ProcessingJob` table (worker/job bookkeeping, not populated by
`ETLWorker` directly in the code read here — likely used by an orchestration layer not
covered in this pass). This is the **queryable lineage** referenced in the platform's
`../ARCHITECTURE.md` ETL tab — one row per video/frame/annotation, so "which frames came
from which video, with what consensus annotations" is a SQL query, not a filesystem scan.

**TFRecords** (`writer.py`) — `TFRecordWriter` writes the *same* feature schema
`datasets/collate.py::_TFRECORD_FEATURE_DESCRIPTION` expects (`image/encoded`,
`image/boxes`, `image/labels`, etc.) — this is the direct hook between ETL output and the
training TFRecord ingestion path. Auto-shards every `shard_size` (default 1000) records
by closing and reopening a new `shard_{video_id}_{NNNNN}.tfrecord` file. Supports
`s3://` output (writes locally to `/tmp/etl_shards` then `boto3` uploads on close, since
`tf.io.TFRecordWriter` can't target S3 directly).

## Config-driven, not hardcoded

Every tunable (stride, scene-change threshold, per-model confidence thresholds, IoU/vote
thresholds, output shard size/location, Ray worker count) comes from one `etl:` config
block, loaded and `${VAR}`-expanded by `cli/etl.py` before being handed to `run_etl`.
