#!/usr/bin/env python3
import sys
import numpy as np
import yaml
from pathlib import Path
from PIL import Image, ImageDraw

# =============================================================================
# CONFIGURE THIS BEFORE RUNNING
# =============================================================================
IMAGE_PATH    = Path("path/to/test_image.jpg")
DEPLOY_CONFIG = Path("configs/deploy/mobilenetv2_ssd_voc_jetson.yaml")
OUTPUT_PATH   = Path("inference_out/onnx_result.jpg")
# =============================================================================

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import onnxruntime as ort

_PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
]


def _iou(box, boxes):
    
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-7)


def nms_numpy(boxes, scores, score_threshold, iou_threshold, max_detections, num_classes):
    
    all_dets = []  # (score, class_id, box)

    for c in range(1, num_classes):  # skip class 0 (background)
        class_scores = scores[:, c]
        keep_mask = class_scores > score_threshold
        if not keep_mask.any():
            continue

        cls_boxes  = boxes[keep_mask]
        cls_scores = class_scores[keep_mask]

        order = np.argsort(cls_scores)[::-1]
        cls_boxes  = cls_boxes[order]
        cls_scores = cls_scores[order]

        suppressed = np.zeros(len(cls_scores), dtype=bool)
        for i in range(len(cls_scores)):
            if suppressed[i]:
                continue
            all_dets.append((cls_scores[i], c, cls_boxes[i]))
            if i + 1 < len(cls_scores):
                ious = _iou(cls_boxes[i], cls_boxes[i + 1:])
                suppressed[i + 1:] |= ious > iou_threshold

    all_dets.sort(key=lambda x: x[0], reverse=True)
    all_dets = all_dets[:max_detections]

    if not all_dets:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    final_scores  = np.array([d[0] for d in all_dets])
    final_classes = np.array([d[1] for d in all_dets], dtype=int)
    final_boxes   = np.stack([d[2] for d in all_dets])
    return final_boxes, final_scores, final_classes


# ---- Load config ----
with open(DEPLOY_CONFIG) as f:
    config = yaml.safe_load(f)

deploy          = config['deploy']
H, W            = deploy['input']['size'][:2]
score_threshold = deploy['post_processing']['score_threshold']
iou_threshold   = deploy['post_processing']['nms_iou_threshold']
max_detections  = deploy['post_processing']['max_detections']
num_classes     = deploy['classes']['num_classes']
onnx_path       = Path(__file__).parent.parent / deploy['onnx_path']

# ---- Load label map ----
label_path = Path(__file__).parent.parent / "datasets/VOCdevkit/labels/voc_labels.txt"
with open(label_path) as f:
    labels = ["background"] + [line.strip() for line in f if line.strip()]

# ---- Load ONNX model ----
print(f"Loading: {onnx_path}")
sess = ort.InferenceSession(str(onnx_path))
input_name = sess.get_inputs()[0].name

# ---- Preprocess image ----
print(f"Image:   {IMAGE_PATH}")
orig_img = Image.open(IMAGE_PATH).convert("RGB")
orig_W, orig_H = orig_img.size

img_arr = np.array(orig_img.resize((W, H)), dtype=np.float32) / 255.0  # norm baked into model
img_arr = img_arr[np.newaxis]  # (1, H, W, 3)

# ---- Run inference ----
raw_outputs = sess.run(None, {input_name: img_arr})
result = {o.name: raw_outputs[i] for i, o in enumerate(sess.get_outputs())}

boxes  = result['boxes'][0]   # (13502, 4) xyxy normalized — already decoded
scores = result['scores'][0]  # (13502, 21) softmax

# ---- NMS ----
final_boxes, final_scores, final_classes = nms_numpy(
    boxes, scores, score_threshold, iou_threshold, max_detections, num_classes
)

# ---- Scale to original image pixel coords ----
scale = np.array([orig_W, orig_H, orig_W, orig_H], dtype=np.float32)
final_boxes_px = (final_boxes * scale).astype(int)

# ---- Draw ----
draw = ImageDraw.Draw(orig_img)
for box, score, class_id in zip(final_boxes_px, final_scores, final_classes):
    label = labels[class_id] if class_id < len(labels) else str(class_id)
    colour = _PALETTE[hash(label) % len(_PALETTE)]
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
    draw.text((x1, max(y1 - 12, 0)), f"{label} {score:.2f}", fill=colour)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
orig_img.save(str(OUTPUT_PATH))

# ---- Print results ----
print(f"\n{len(final_boxes)} detections:")
for box, score, class_id in zip(final_boxes, final_scores, final_classes):
    label = labels[class_id] if class_id < len(labels) else str(class_id)
    print(f"  {label:<15} {score:.3f}  {tuple(np.round(box, 3))}")
print(f"\nSaved → {OUTPUT_PATH}")
