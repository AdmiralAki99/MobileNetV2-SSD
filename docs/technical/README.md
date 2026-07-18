# Technical Reference Docs

Component-level reference documentation for this repo, meant to answer "what does this
part do and how does it fit together" months after you last touched it. For the bigger
picture of the ml-platform this repo submodules into, see [../ARCHITECTURE.md](../ARCHITECTURE.md).

| Doc | Covers |
|---|---|
| [repo-map.md](repo-map.md) | Repo layout, train→export→inference data flow, where this submodule sits in the platform |
| [model-backbone.md](model-backbone.md) | MobileNetV2 backbone: blocks, ImageNet weight transplant, feature maps |
| [model-ssd.md](model-ssd.md) | SSD head: FPN, anchors/priors, box ops, matching, loss, post-processing/NMS |
| [training.md](training.md) | Training engine, optimizer/schedule, AMP, checkpoints, EMA |
| [datasets.md](datasets.md) | Dataset loading (VOC, VisDrone), transforms, collate |
| [deploy.md](deploy.md) | export → convert (ONNX) → quantize → validate pipeline |
| [etl.md](etl.md) | Video → frame → detection → consensus ETL pipeline |
| [infrastructure.md](infrastructure.md) | DynamoDB experiment ledger, S3 sync |
| [cli.md](cli.md) | CLI entrypoints (train, inference, bundle, etl) |
