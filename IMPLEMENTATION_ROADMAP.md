
## Implementation Roadmap


# 📦 MobileNetV2-SSD Project Roadmap

> Centralized roadmap for development, testing, and deployment  
> Use ✅ for done, 🚧 for in progress, 📝 for planned

---

## 🧱 1. Core Infrastructure

### 📁 Repo Setup
- [✅] `pyproject.toml` / `requirements.txt`
- [✅] `Makefile` (train, test, format)
- [ ] `.pre-commit-config.yaml`
- [ ] `.github/workflows/ci.yml`
- [✅] Folder structure finalized
- [✅] Config loading (`core/config.py`)
- [ ] Logging / utils / profiler
- [📝] Distributed training scaffolding (optional later)

---

## 📊 2. Datasets & Transforms

### 📂 Datasets
- [ ] `datasets/base.py` (abstract)
- [ ] `datasets/voc.py`
- [📝] `datasets/coco.py`
- [🚧] `datasets/transforms_tf.py` (augmentations for tf.data)
- [ ] `datasets/collate.py`
- [📝] `datasets/cache.py` (optional speed-up)

### 🎛️ Orchestration
- [📝] `orchestration/data_orch.py`  
  → Build train/val/test pipelines

---

## 🧠 3. Model Components

### 🧩 MobileNetV2 Backbone
- [✅] `mobilenet_v2/blocks.py`
- [✅] `mobilenet_v2/backbone.py`

### 🧮 SSD Heads & Utils
- [x] `ssd/ops/box_ops_tf.py`
- [x] `ssd/ops/encode_ops_tf.py`
- [x] `ssd/ops/heads_tf.py`
- [x] `ssd/ops/loss_ops_tf.py`
- [x] `ssd/ops/match_ops_tf.py`
- [x] `ssd/ops/postprocess_tf.py`
- [x] `ssd/orchestration/conf_loss_orch.py`
- [x] `ssd/orchestration/hard_neg_orch.py`
- [x] `ssd/orchestration/loss_orch.py`
- [x] `ssd/orchestration/priors_orch.py`
- [x] `ssd/orchestration/targets_orch.py`
- [x] `ssd/fpn.py` (Feature Pyramid Network)
- [x] `ssd/model.py` ( SSD Model Creation)

- [✅] `factory.py` (Model Factory Pattern)

---

## ⚙️ 4. Orchestrations (High-Level “Recipes”)

### 📐 Geometry & Anchors
- [✅] `orchestration/priors_orch.py`  
  → Builds priors grid from config

### 🎯 Targets & Matching
- [✅] `orchestration/targets_orch.py`  
  → Match GTs to priors, encode offsets

### ⚖️ Loss & HNM
- [✅] `orchestration/loss_orch.py`  
  → Combines cls/loc losses + normalization
- [✅] `orchestration/conf_loss.py`  
  → Per anchor loss for the predictions
- [✅] `orchestration/hard_neg_orch.py`  
  → Select negatives via OHEM ratio

---

## 🚀 5. Training Subsystem

### 🔁 Core
- [✅] `training/engine.py` (train_one_epoch, evaluate, fit)
	- [x] `training_step` (Training Over a batch)
	- [x] `train_one_epoch` (Training over one epoch)
	- [ ] `evaluate` (Evaluate over validation dataset)
	- [ ] `fit` (Train over epochs and evaluate after every epoch)
- [ ] `training/optimizer.py`
- [x] `training/scheduler.py`
- [x] `training/checkpoints.py`
- [x] `training/ema.py`
- [ ] `training/amp.py`

### 📊 Metrics
- [✅] `training/metrics.py` (VOC mAP@0.5)
- [📝] COCO-style mAP (optional later)

### 🧩 Orchestration Integration
- [🚧] Replace direct ops with orchestration calls (priors → targets → loss)

---

## 🧪 6. Tests & Validation

### ✅ Unit Tests
- [✅] `test_box_ops_tf.py`
- [✅] `test_anchors_tf.py`
- [✅] `test_losses_tf.py`
- [🚧] `test_matcher_tf.py`
- [🚧] `test_postprocess_tf.py`
- [🚧] `test_targets_orch.py`
- [🚧] `test_loss_orch.py`
- [🚧] `test_hard_neg_orch.py`

### 🧩 Integration Tests
- [📝] Synthetic batch end-to-end (priors → match → loss → grad)
- [📝] Decode+NMS output parity (TF vs NumPy)

---

## ☁️ 7. Cloud Training (AWS)

- [📝] `docker/train.Dockerfile` (TF + deps)
- [📝] `k8s/train-job.yaml`
- [📝] Checkpoint → S3 syncing
- [📝] Logging via TensorBoard / W&B
- [📝] Optional distributed strategy support

---

## 🛰️ 8. Inference & Runtime

### 💻 Desktop / Validation
- [✅] `inference/predictor.py`
- [📝] `inference/export.py`
- [📝] `inference/profiling.py`
- [✅] `inference/postprocess_np.py`

### 🧠 Hailo / Jetson
- [✅] `hailo/preprocessing.py`
- [✅] `hailo/postprocessing.py`
- [📝] `hailo/compile_hailo.py`
- [📝] `hailo/runtime.py`
- [📝] TensorRT build script (Jetson)

---

## 🕹️ 9. Drone Runtime

- [🚧] `drone/camera.py`
- [🚧] `drone/streamer.py`
- [🚧] `drone/node.py`  
  → capture → infer → overlay → transmit
- [📝] `orchestration/drone_node_orch.py`  
  → runtime glue loop (camera + inference)

---

## 🧰 10. CLI & Scripts

- [🚧] `cli/train.py`
- [🚧] `cli/eval.py`
- [🚧] `cli/infer.py`
- [📝] `cli/export.py`
- [📝] `cli/visualize_anchors.py`
- [🚧] Dataset prep scripts (`prepare_voc.sh`, etc.)
- [📝] `scripts/package_runtime.py` (build deployable bundle)

---

## 📈 11. Deployment Pipeline

- [📝] Cloud → Engine Export → Device Bundle
- [📝] Version manifest / checksum
- [📝] OTA update script (optional later)
- [📝] Device pulls minimal runtime only

---

## 🧩 12. Optional Future Enhancements

- [📝] PyTorch backend (mirror TF ops)
- [📝] Multi-scale training
- [📝] Mosaic / MixUp augmentation
- [📝] Quantization-aware training
- [📝] Semi-supervised fine-tuning module

---

### Legend
| Symbol | Meaning     |
| :----: | :---------- |
|   ✅    | Completed   |
|   🚧   | In progress |
|   📝   | Planned     |

---

**Notes:**
- Core focus first → `ops` + `orchestration` + `training`.
- Cloud & deployment can be added incrementally.
- The orchestrations are your glue layer: everything above them can run locally or in the cloud identically.
