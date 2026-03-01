import argparse
import traceback
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

from deploy import load_deploy_config
from mobilenetv2ssd.core.config import PROJECT_ROOT

_PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a MobileNetV2 SSD ONNX model.")
    parser.add_argument('--deploy_config', type=str, required=True, help='Path to the deployment configuration file.')
    parser.add_argument('--model', choices=['fp32', 'int8'], default='fp32',
                        help='Which ONNX model to use: fp32 (default) or int8 (quantized).')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image', type=str, help='Path to an image file or directory of images.')
    mode.add_argument('--webcam', action='store_true', help='Run live inference from webcam.')

    parser.add_argument('--camera', type=str, default='0',
                        help='Camera source: device index (default 0) or HTTP stream URL.')
    parser.add_argument('--output_dir', type=str, default='inference_out/',
                        help='Directory to save annotated images (image mode only).')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Export directory containing model.onnx / model_int8.onnx. Overrides deploy config paths.')

    args = parser.parse_args()
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera

    return {
        'deploy_config': Path(args.deploy_config),
        'model': args.model,
        'image': Path(args.image) if args.image else None,
        'webcam': args.webcam,
        'camera': camera_source,
        'output_dir': Path(args.output_dir),
        'model_dir': Path(args.model_dir) if args.model_dir else None,
    }


def load_label_map(deploy_config: dict[str, Any]) -> list[str]:
    root_str = deploy_config['deploy']['classes']['root']

    if root_str.startswith("${"):
        label_map_path = PROJECT_ROOT / "datasets/VOCdevkit/labels/voc_labels.txt"
    else:
        label_map_path = Path(root_str)

    with open(label_map_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    if not deploy_config['deploy']['classes']['use_sigmoid']:
        lines = ["background"] + lines  # class 0 is background for softmax

    return lines


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_box + area_boxes - inter + 1e-7)


def run_nms(boxes: np.ndarray, scores: np.ndarray, deploy_config: dict[str, Any]):
    pp             = deploy_config['deploy']['post_processing']
    score_thresh   = pp['score_threshold']
    iou_thresh     = pp['nms_iou_threshold']
    max_detections = pp['max_detections']
    num_classes    = deploy_config['deploy']['classes']['num_classes']

    all_dets = []
    for c in range(1, num_classes):  # skip background (class 0)
        class_scores = scores[:, c]
        keep = class_scores > score_thresh
        if not keep.any():
            continue

        cls_boxes  = boxes[keep]
        cls_scores = class_scores[keep]

        order      = np.argsort(cls_scores)[::-1]
        cls_boxes  = cls_boxes[order]
        cls_scores = cls_scores[order]

        suppressed = np.zeros(len(cls_scores), dtype=bool)
        for i in range(len(cls_scores)):
            if suppressed[i]:
                continue
            all_dets.append((cls_scores[i], c, cls_boxes[i]))
            if i + 1 < len(cls_scores):
                ious = _iou(cls_boxes[i], cls_boxes[i + 1:])
                suppressed[i + 1:] |= ious > iou_thresh

    all_dets.sort(key=lambda x: x[0], reverse=True)
    all_dets = all_dets[:max_detections]

    if not all_dets:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    final_scores  = np.array([d[0] for d in all_dets])
    final_classes = np.array([d[1] for d in all_dets], dtype=int)
    final_boxes   = np.stack([d[2] for d in all_dets])
    return final_boxes, final_scores, final_classes


def preprocess_image(image_path: Path, H: int, W: int):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img.resize((W, H)), dtype=np.float32) / 255.0
    return img, arr[np.newaxis]  # original PIL image + (1,H,W,3) array


def preprocess_frame(frame_bgr: np.ndarray, H: int, W: int) -> np.ndarray:
    frame_rgb = frame_bgr[:, :, ::-1]
    img = Image.fromarray(frame_rgb).resize((W, H))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis]  # (1,H,W,3)


def draw_detections(img: Image.Image, boxes_norm: np.ndarray, scores: np.ndarray,
                    class_ids: np.ndarray, labels: list[str]) -> Image.Image:
    orig_W, orig_H = img.size
    scale = np.array([orig_W, orig_H, orig_W, orig_H], dtype=np.float32)
    boxes_px = (boxes_norm * scale).astype(int)

    draw = ImageDraw.Draw(img)
    for box, score, class_id in zip(boxes_px, scores, class_ids):
        label  = labels[class_id] if class_id < len(labels) else str(class_id)
        colour = _PALETTE[hash(label) % len(_PALETTE)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        draw.text((x1, max(y1 - 12, 0)), f"{label} {score:.2f}", fill=colour)

    return img


def draw_detections_cv2(frame: np.ndarray, boxes_norm: np.ndarray, scores: np.ndarray,
                        class_ids: np.ndarray, labels: list[str]) -> np.ndarray:
    import cv2

    orig_H, orig_W = frame.shape[:2]
    scale    = np.array([orig_W, orig_H, orig_W, orig_H], dtype=np.float32)
    boxes_px = (boxes_norm * scale).astype(int)

    for box, score, class_id in zip(boxes_px, scores, class_ids):
        label      = labels[class_id] if class_id < len(labels) else str(class_id)
        r, g, b    = _PALETTE[hash(label) % len(_PALETTE)]
        colour_bgr = (b, g, r)
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour_bgr, 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr, 1)

    return frame


def run_webcam(sess: ort.InferenceSession, input_name: str, labels: list[str],
               deploy_config: dict[str, Any], H: int, W: int, camera_source) -> int:
    import cv2
    import time

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return 1

    print("Webcam running — press 'q' to quit.")
    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            img_arr = preprocess_frame(frame, H, W)

            raw_outputs = sess.run(None, {input_name: img_arr})
            result      = {o.name: raw_outputs[i] for i, o in enumerate(sess.get_outputs())}

            boxes  = result['boxes'][0]
            scores = result['scores'][0]

            final_boxes, final_scores, final_classes = run_nms(boxes, scores, deploy_config)
            frame = draw_detections_cv2(frame, final_boxes, final_scores, final_classes, labels)

            now  = time.perf_counter()
            fps  = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("MobileNetV2 SSD (ONNX)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def execute_inference() -> int:
    try:
        args = parse_args()

        deploy_config = load_deploy_config(args['deploy_config'])
        H, W = deploy_config['deploy']['input']['size'][:2]

        if args['model_dir']:
            onnx_path = args['model_dir'] / ("model_int8.onnx" if args['model'] == 'int8' else "model.onnx")
        else:
            onnx_key  = 'quantized_onnx_path' if args['model'] == 'int8' else 'onnx_path'
            onnx_path = PROJECT_ROOT / deploy_config['deploy'][onnx_key]

        print(f"Model  : {onnx_path}")
        sess       = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name

        labels = load_label_map(deploy_config)

        if args['webcam']:
            return run_webcam(sess, input_name, labels, deploy_config, H, W, args['camera'])

        image_arg = args['image']
        if image_arg.is_dir():
            extensions  = ("*.jpg", "*.jpeg", "*.png")
            image_paths = sorted(p for ext in extensions for p in image_arg.glob(ext))
        else:
            image_paths = [image_arg]

        output_dir = args['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            orig_img, img_arr = preprocess_image(image_path, H, W)

            raw_outputs = sess.run(None, {input_name: img_arr})
            result      = {o.name: raw_outputs[i] for i, o in enumerate(sess.get_outputs())}

            boxes  = result['boxes'][0]
            scores = result['scores'][0]

            final_boxes, final_scores, final_classes = run_nms(boxes, scores, deploy_config)
            annotated = draw_detections(orig_img, final_boxes, final_scores, final_classes, labels)

            out_path = output_dir / image_path.name
            annotated.save(str(out_path))

            det_summary = [
                (labels[c] if c < len(labels) else str(c), f"{s:.2f}")
                for c, s in zip(final_classes, final_scores)
            ]
            print(f"{image_path.name}  {len(final_boxes)} detections: {det_summary}")
            print(f"Saved → {out_path}")

        return 0

    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(execute_inference())
