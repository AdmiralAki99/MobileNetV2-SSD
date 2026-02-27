import argparse
import traceback
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from deploy import load_deploy_config
from mobilenetv2ssd.core.config import PROJECT_ROOT
from mobilenetv2ssd.models.ssd.ops.postprocess_tf import (
    _prepare_nms_inputs, _run_batched_nms, _restore_to_image_space,
)

_PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a MobileNetV2 SSD SavedModel.")
    parser.add_argument('--deploy_config', type=str, required=True, help='Path to the deployment configuration file.')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image', type=str, help='Path to image file or directory of images.')
    mode.add_argument('--webcam', action='store_true', help='Run live inference from webcam.')

    parser.add_argument('--camera', type=str, default='0',
                        help='Camera source: device index (default 0) or HTTP stream URL '
                             '(e.g. http://192.168.x.x:8080/live for IP Camera Lite).')
    parser.add_argument('--output_dir', type=str, default='inference_out/', help='Directory to save annotated images (image mode only).')

    args = parser.parse_args()

    # Camera source: integer index or URL string
    camera_arg = args.camera
    camera_source = int(camera_arg) if camera_arg.isdigit() else camera_arg

    return {
        'deploy_config': Path(args.deploy_config),
        'image': Path(args.image) if args.image else None,
        'webcam': args.webcam,
        'camera': camera_source,
        'output_dir': Path(args.output_dir),
    }


def load_label_map(deploy_config: dict[str, Any]):
    root_str = deploy_config['deploy']['classes']['root']

    
    if root_str.startswith("${"):
        label_map_path = PROJECT_ROOT / "datasets/VOCdevkit/labels/voc_labels.txt"
    else:
        label_map_path = Path(root_str)

    with open(label_map_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    use_sigmoid = deploy_config['deploy']['classes']['use_sigmoid']
    if not use_sigmoid:
        lines = ["background"] + lines  # class 0 is background for softmax

    return lines


def preprocess_image(image_path: Path, H: int, W: int):
    img = Image.open(image_path).convert("RGB").resize((W, H))
    arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1] normalized boxes baked into the model
    return tf.expand_dims(tf.constant(arr), axis=0)


def preprocess_frame(frame_bgr: np.ndarray, H: int, W: int):

    frame_rgb = frame_bgr[:, :, ::-1]  # BGR → RGB
    img = Image.fromarray(frame_rgb).resize((W, H))
    arr = np.array(img, dtype=np.float32) / 255.0  # norm baked into SavedModel
    return tf.expand_dims(tf.constant(arr), axis=0)


def run_nms(boxes_xyxy: tf.Tensor, scores: tf.Tensor, deploy_config: dict[str, Any], orig_H: int, orig_W: int):
    pp = deploy_config['deploy']['post_processing']

    # SavedModel outputs xyxy (NMS expects yxyx)
    x1, y1, x2, y2 = tf.split(boxes_xyxy, num_or_size_splits=4, axis=-1)
    boxes_yxyx = tf.concat([y1, x1, y2, x2], axis=-1)

    # Strip background (class 0) before NMS
    scores_no_bg = scores[:, :, 1:]

    nms_boxes, nms_scores = _prepare_nms_inputs(boxes_yxyx, scores_no_bg)
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = _run_batched_nms(
        nms_boxes, nms_scores,
        iou_thresh=pp['nms_iou_threshold'],
        scores_thresh=pp['score_threshold'],
        top_k=pp['per_class_top_k'],
        max_detections=pp['max_detections'],
    )

    n = int(valid_detections[0].numpy())
    nmsed_boxes   = nmsed_boxes[0, :n]
    nmsed_scores  = nmsed_scores[0, :n]
    
    # Class IDs add 1 to restore original
    nmsed_classes = nmsed_classes[0, :n].numpy().astype(int) + 1

    # Scale to original image pixel coords — returns xyxy
    boxes_px = _restore_to_image_space(nmsed_boxes, orig_H, orig_W)

    return (
        boxes_px.numpy().astype(int),
        nmsed_scores.numpy(),
        nmsed_classes,
    )


def draw_detections(image_path: Path, boxes, scores, class_ids, labels):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for box, score, class_id in zip(boxes, scores, class_ids):
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        colour = _PALETTE[hash(label) % len(_PALETTE)]
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        draw.text((x1, max(y1 - 12, 0)), f"{label} {score:.2f}", fill=colour)

    return img


def draw_detections_cv2(frame: np.ndarray, boxes, scores, class_ids, labels):

    import cv2

    for box, score, class_id in zip(boxes, scores, class_ids):
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        r, g, b = _PALETTE[hash(label) % len(_PALETTE)]
        colour_bgr = (b, g, r)  # OpenCV uses BGR
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour_bgr, 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_bgr, 1)

    return frame


def run_webcam(infer, labels: list[str], deploy_config: dict[str, Any], H: int, W: int, camera_source: int | str = 0):
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

            orig_H, orig_W = frame.shape[:2]

            # Resize to model input size for inference; draw on full-resolution frame
            img_tensor = preprocess_frame(frame, H, W)
            result = infer(input_image=img_tensor)

            boxes_px, det_scores, class_ids = run_nms(
                result['boxes'], result['scores'], deploy_config, orig_H, orig_W
            )

            frame = draw_detections_cv2(frame, boxes_px, det_scores, class_ids, labels)

            # FPS overlay
            now = time.perf_counter()
            fps = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("MobileNetV2 SSD", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def execute_inference():
    try:
        args = parse_args()

        # Allow TF to grow GPU memory on demand instead of reserving it all upfront
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        deploy_config = load_deploy_config(args['deploy_config'])
        H, W = deploy_config['deploy']['input']['size'][:2]
        saved_model_path = PROJECT_ROOT / deploy_config['deploy']['saved_model_path']

        model = tf.saved_model.load(str(saved_model_path))
        infer = model.signatures["serving_default"]

        labels = load_label_map(deploy_config)

        # --- Webcam mode ---
        if args['webcam']:
            return run_webcam(infer, labels, deploy_config, H, W, args['camera'])

        # --- Image / directory mode ---
        image_arg = args['image']
        if image_arg.is_dir():
            image_paths = sorted(image_arg.glob("*.jpg")) + sorted(image_arg.glob("*.png"))
        else:
            image_paths = [image_arg]

        output_dir = args['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            orig_img = Image.open(image_path)
            orig_W, orig_H = orig_img.size
            orig_img.close()

            img_tensor = preprocess_image(image_path, H, W)
            result = infer(input_image=img_tensor)

            boxes_px, det_scores, class_ids = run_nms(
                result['boxes'], result['scores'], deploy_config, orig_H, orig_W
            )

            annotated = draw_detections(image_path, boxes_px, det_scores, class_ids, labels)
            out_path = output_dir / image_path.name
            annotated.save(str(out_path))

            det_summary = [
                (labels[c] if c < len(labels) else str(c), f"{s:.2f}")
                for c, s in zip(class_ids, det_scores)
            ]
            print(f"{image_path.name}  {len(boxes_px)} detections: {det_summary}")
            print(f"Saved → {out_path}")

        return 0

    except Exception:
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(execute_inference())
