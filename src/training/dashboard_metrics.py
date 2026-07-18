import json
from pathlib import Path
import numpy as np

from .metrics import _box_iou_xyxy


class DashboardMetricsAccumulator:
    def __init__(self):

        # AP per class
        self.class_ap = {}

        self.confidence_matrix = []
        self.images = []
        self.train_loss = []
        self.map_curve = []
        self.learning_curve = []
        self.nms_mean_score = []
        self.nms_avg_det = []
        self.nms_zero_det = []

    def append_epoch(self, train_loss, map_score, learning_rate, nms_mean_score, nms_avg_det, nms_zero_det):

        self.train_loss.append(float(train_loss))
        self.map_curve.append(float(map_score))
        self.learning_curve.append(float(learning_rate))
        self.nms_mean_score.append(float(nms_mean_score))
        self.nms_avg_det.append(float(nms_avg_det))
        self.nms_zero_det.append(float(nms_zero_det))

    def set_class_ap(self, class_ap: dict):

        self.class_ap = {name: float(value) for name, value in class_ap.items()}

    def set_confusion_matrix(self, matrix: np.ndarray):
        self.confidence_matrix = matrix.astype(int).tolist()

    def set_sample_images(self, images: list[dict]):
        self.images = images

    def to_dict(self):

        return {
            "train_loss": self.train_loss,
            "map_curve": self.map_curve,
            "lr_curve": self.learning_curve,
            "nms_mean_score": self.nms_mean_score,
            "nms_avg_det": self.nms_avg_det,
            "nms_zero_det": self.nms_zero_det,
            "class_ap": self.class_ap,
            "conf_mat": self.confidence_matrix,
            "images": self.images,
        }

    def save(self, path: Path):
        data = json.dumps(self.to_dict())
        path.write_text(data, encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        # Factory method at a class level to create these dashboard metrics
        if not path.exists():
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        accumulator = cls()

        accumulator.confidence_matrix = data.get("conf_mat", [])
        accumulator.images = data.get("images", [])
        accumulator.train_loss = data.get("train_loss", [])
        accumulator.map_curve = data.get("map_curve", [])
        accumulator.learning_curve = data.get("lr_curve", [])
        accumulator.nms_mean_score = data.get("nms_mean_score", [])
        accumulator.nms_avg_det = data.get("nms_avg_det", [])
        accumulator.nms_zero_det = data.get("nms_zero_det", [])
        accumulator.class_ap = data.get("class_ap", {})

        return accumulator


def build_confusion_matrix(predictions, ground_truth, num_classes, iou_thresh=0.5):
    gt_by_image = {gt_info["image_id"]: gt_info for gt_info in ground_truth}
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for prediction in predictions:
        image_id = prediction["image_id"]

        if image_id not in gt_by_image:
            gt_boxes, gt_labels = np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)
        else:
            gt_boxes, gt_labels = gt_by_image[image_id]["boxes"], gt_by_image[image_id]["labels"]

        matched = np.zeros(len(gt_boxes), dtype=bool)
        order = np.argsort(-prediction["scores"])

        for index in order:
            label = int(prediction["labels"][index])
            if label == 0:
                continue

            if len(gt_boxes) == 0:
                # This is a false positive from the model
                matrix[0, label] = matrix[0, label] + 1
                continue

            # Computing the IoU between this detection and every GT box for the image
            iou_matrix = _box_iou_xyxy(prediction["boxes"][index], gt_boxes)

            # Taking the best IoU
            best_index = int(np.argmax(iou_matrix))
            best_iou = iou_matrix[best_index]

            if best_iou >= iou_thresh and not matched[best_index]:
                matched[best_index] = True
                matrix[int(gt_labels[best_index]), label] = matrix[int(gt_labels[best_index]), label] + 1
            else:
                matrix[0, label] = matrix[0, label] + 1

        for gt_index in range(len(matched)):
            if not matched[gt_index]:
                matrix[int(gt_labels[gt_index]), 0] = matrix[int(gt_labels[gt_index]), 0] + 1

    return matrix


def sample_detection_images(predictions, image_size, class_names, max_images=8, score_threshold=0.3):
    H, W = image_size
    images = []

    for prediction in predictions:
        boxes = []
        for box, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
            label = int(label)
            if label == 0 or score < score_threshold:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]
            boxes.append(
                {
                    "x": max(x1 / W, 0.0),
                    "y": max(y1 / H, 0.0),
                    "w": max((x2 - x1) / W, 0.0),
                    "h": max((y2 - y1) / H, 0.0),
                    "cls": class_names.get(label, f"class_{label}"),
                    "score": float(score),
                }
            )

        if not boxes:
            continue

        images.append(
            {
                "id": prediction["image_id"],
                "label": boxes[0]["cls"],
                "boxes": boxes,
            }
        )

        if len(images) >= max_images:
            break

    return images
