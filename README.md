# MobileNetV2-SSD – Edge-Oriented Object Detection

This repository contains an in-progress implementation of a MobileNetV2-SSD
object detection pipeline designed for real-time inference on embedded hardware
(e.g., Jetson, Hailo).

## Project Goals
- Build an SSD detection system from first principles
- Emphasize modular design and reproducibility
- Support both training and edge deployment
- Integrate with downstream robotics/drone runtime

## Current Status
This project is under active development. Core SSD components, loss
orchestration, and inference paths are implemented. Training, evaluation,
and deployment tooling are being added incrementally.

See `IMPLEMENTATION_ROADMAP.md` for a detailed breakdown of completed,
in-progress, and planned components.

## Key Implemented Components
- MobileNetV2 backbone
- SSD heads and priors
- Matching and encoding logic
- Hard negative mining
- Modular loss orchestration
- Initial inference pipeline

## Component Validation (Notebooks)
This project uses a "Notebook-Driven Development" approach. Each subsystem was implemented and verified in isolation before being promoted to the `src/` codebase.

**1. Core Primitives (01-08)**
* `01_backbone.ipynb` - `08_heads.ipynb`: Unit verification of the MobileNetV2 feature extractor, bounding box encoding logic, and SSD head operations.

**2. The Orchestration Layer (09-13)**
* `09_priors_orch.ipynb` & `10_targets_orch.ipynb`: Visualizing anchor grid generation and ground-truth matching.
* `11_hard_neg_orch.ipynb` - `13_loss_orch.ipynb`: Verification of the Hard Negative Mining ratio (3:1) and MultiBox Loss convergence.

**3. System Integration (14-18)**
* `14_factory_ssd.ipynb`: Testing the Model Factory pattern.
* `17_Scheduler.ipynb` & `18_Checkpoint_Manager.ipynb`: Validating the learning rate decay schedule and state preservation logic.

## Planned Work
- Validation/evaluation loop
- EMA and AMP training
- Deployment packaging for Jetson/Hailo
- ROS2 runtime integration

## Notes
This repository reflects an ongoing engineering effort rather than a finished
product. Design decisions prioritize clarity, modularity, and real-world
deployment constraints.
