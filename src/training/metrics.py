from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
import tensorflow as tf

@dataclass
class Detection:
    image_id: str
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    
@dataclass
class GroundTruth:
    image_id: str
    boxes: np.ndarray
    labels: np.ndarray
    
@dataclass
class ClassGroundTruth:
    boxes: np.ndarray
    detected: np.ndarray
    
    
class MeanAveragePrecision:
    def __init__(self, num_classes: int, iou_thresholds: list[float] | float = 0.5, style: Literal["voc", "coco"] = "voc", class_names: dict[int, str] | None = None):
        
        self.num_classes = num_classes
        self.style = style
        self.class_names = class_names or {}

        if isinstance(iou_thresholds, (int, float)):
            self.iou_thresholds = [float(iou_thresholds)]
        else:
            self.iou_thresholds = [float(thresh) for thresh in iou_thresholds]

        self._predictions: list[Detection] = []
        self._ground_truths: list[GroundTruth] = []

    def reset(self):
        self._predictions.clear()
        self._ground_truths.clear()

    def update(self, predictions: list[dict], ground_truths: list[dict]):
        for prediction in predictions:
            self._predictions.append(Detection(
                image_id = self._to_str(prediction['image_id']),
                boxes = np.asarray(prediction['boxes'], dtype = np.float32),
                scores = np.asarray(prediction['scores'], dtype = np.float32),
                labels = np.asarray(prediction['labels'], dtype = np.float32)
            ))
            
        for ground_truth in ground_truths:
            self._ground_truths.append(GroundTruth(
                image_id = self._to_str(ground_truth['image_id']),
                boxes = np.asarray(ground_truth['boxes'], dtype = np.float32),
                labels = np.asarray(ground_truth['labels'], dtype = np.float32)
            ))

    @staticmethod
    def _to_str(val):
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)

    def _organize_ground_truths(self):
        
        by_class : dict[int, dict[str, GroundTruth]] = {}

        for ground_truth in self._ground_truths:
            for box, label in zip(ground_truth.boxes, ground_truth.labels):
                class_num = int(label)

                if class_num not in by_class:
                    # Needs to be added
                    by_class[class_num] = {}

                # Now checking if the image is present in the class detections
                if ground_truth.image_id not in by_class[class_num]:
                    # Creating the place holders
                    by_class[class_num][ground_truth.image_id] = ClassGroundTruth(
                        boxes = np.empty((0,4), dtype = np.float32),
                        detected = np.empty((0,), dtype= bool)
                    )

                # It exists in the list by now
                class_gt = by_class[class_num][ground_truth.image_id]
                by_class[class_num][ground_truth.image_id] = ClassGroundTruth(
                    boxes = np.vstack([class_gt.boxes, box.reshape(1,4)]),
                    detected = np.append(class_gt.detected, False)
                )

        return by_class

    def _count_positives(self, ground_truth_by_class: dict):

        positive_counts = {class_num: 0 for class_num in range(self.num_classes)}

        for class_index, images in ground_truth_by_class.items():
            for class_gt in images.values():
                positive_counts[class_index] = positive_counts[class_index] + len(class_gt.boxes)
        return positive_counts

    def _organize_predictions(self):

        preds_by_class: dict[int, list[dict]] = {}

        for detection in self._predictions:
            for box, score, label in zip(detection.boxes, detection.scores, detection.labels):
                class_index = int(label)
                # Checking if the detection is a background
                if class_index == 0:
                    continue

                # Creating predictions by class
                if class_index not in preds_by_class:
                    preds_by_class[class_index] = []

                preds_by_class[class_index].append({
                    'image_id': detection.image_id,
                    'box': box,
                    'score': float(score)
                })

        # Sorting by class index for easier debugging
        for class_index in preds_by_class:
            preds_by_class[class_index].sort(key = lambda x: x['score'], reverse = True)
        
        return preds_by_class

    @staticmethod
    def _ap_101_point(recall: np.ndarray, precision:np.ndarry):
        if len(recall) == 0:
            return 0.0

        recall_pts = np.linspace(0.0, 1.0, 101)
        precision_inter = np.zeros_like(recall_pts)

        for index, recall_pt in enumerate(recall_pts):
            valid_mask = recall >= recall_pt
            if np.any(valid_mask):
                precision_inter[index] = np.max(precision[valid_mask])

        return float(np.mean(precision_inter))
    
    @staticmethod
    def _ap_voc(recall: np.ndarray, precision: np.ndarray):

        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precision, [0.0]])

        for index in range(len(mpre) - 1, 0, -1):
            mpre[index - 1] = max(mpre[index - 1], mpre[index])

        index = np.where(mrec[1:] != mrec[:-1])[0]

        AP = np.sum((mrec[index + 1] - mrec[index]) * mpre[index + 1])
        return float(AP)

    def _format_results(self, AP: np.ndarray, num_positives: dict[int,int]):
        results = {}
    
        valid_classes = [class_num for class_num in range(1, self.num_classes) if num_positives[class_num] > 0]

        if not valid_classes:
            for thresh in self.iou_thresholds:
                results[f"mAP@{thresh:.2f}"] = 0.0
            return results

        AP_valid = AP[valid_classes, :]

        if self.style == "coco" and len(self.iou_thresholds) > 1:
            mAP = float(np.mean(AP_valid))

            threshold_start, threshold_end = self.iou_thresholds[0], self.iou_thresholds[-1]
            results[f"mAP@[{threshold_start:.2f}:{threshold_end:.2f}]"] = mAP

            if 0.5 in self.iou_thresholds:
                index = self.iou_thresholds.index(0.5)
                results["AP@0.50"] = float(np.mean(AP_valid[:, index]))
            if 0.75 in self.iou_thresholds:
                index = self.iou_thresholds.index(0.75)
                results["AP@0.75"] = float(np.mean(AP_valid[:, index]))
        else:
            for iou_index, iou_threshold in enumerate(self.iou_thresholds):
                mAP = float(np.mean(AP_valid[:,iou_index]))
                results[f"mAP@{iou_threshold:.2f}"] = mAP

        for class_index in valid_classes:
            class_name = self.class_names.get(class_index, f"class_{class_index}")
            results[f"AP/{class_name}"] = float(AP[class_index, 0])

        return results   

    def _compute_ap_for_class(self, predictions: list[dict], ground_truths: dict[str, ClassGroundTruth], num_positives: int, iou_threshold: float):

        detected = {
            image_id: np.zeros(len(class_gt.boxes), dtype = np.float32) for image_id, class_gt in ground_truths.items()
        }

        num_preds = len(predictions)

        TP = np.zeros(num_preds, dtype = np.float32)
        FP = np.zeros(num_preds, dtype = np.float32)

        for index, prediction in enumerate(predictions):
            image_id, box = prediction['image_id'], prediction['box']

            if image_id not in ground_truths:
                # If not predicted then its a false positive
                FP[index] = 1.0
                continue

            ground_truth_boxes = ground_truths[image_id].boxes
            detection_flags = detected[image_id]

            if len(ground_truth_boxes) == 0:
                FP[index] = 1.0
                continue

            ious = _box_iou_xyxy(box, ground_truth_boxes)
            best_index = int(np.argmax(ious))
            best_iou = ious[best_index]

            # Need to check if the IoU is more than threshold
            if best_iou >= iou_threshold and not detection_flags[best_index]:
                TP[index] = 1.0
                detection_flags[best_index] = True
            else:
                FP[index] = 1.0

        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)

        recall = TP_cumsum / num_positives
        precision = TP_cumsum/ np.maximum(TP_cumsum + FP_cumsum, 1e-6)

        if self.style == "coco":
            return self._ap_101_point(recall, precision)
        else:
            return self._ap_voc(recall,precision)

    def compute(self):
        if not self._ground_truths:
            return {f"mAP@{thresh:.2f}": 0.0 for thresh in self.iou_thresholds}

        # Calculating the ground truths by class based
        ground_truth_by_class = self._organize_ground_truths()

        num_positives = self._count_positives(ground_truth_by_class)

        # Organizing the prediction scores
        preds_by_class = self._organize_predictions()

        AP = np.zeros((self.num_classes, len(self.iou_thresholds)), dtype = np.float32)

        for class_index in range(1, self.num_classes):
            if num_positives[class_index] == 0:
                continue
                
            class_predictions = preds_by_class.get(class_index, [])
            if not class_predictions:
                continue

            class_gt = ground_truth_by_class.get(class_index, {})

            for iou_index, iou_threshold in enumerate(self.iou_thresholds):
                ap = self._compute_ap_for_class(predictions = class_predictions, ground_truths = class_gt, num_positives = num_positives[class_index], iou_threshold = iou_threshold)
                AP[class_index,iou_index] = ap

        return self._format_results(AP, num_positives)
  
class MetricsCollection:
    def __init__(self, metrics: dict[str, MeanAveragePrecision]):
        self.metrics = metrics

    def update(self, predictions: list[dict], ground_truths: list[dict]):
        for metric in self.metrics.values():
            metric.update(predictions, ground_truths)

    def compute(self):
        results = {}
        for name, metric in self.metrics.items():
            for key, value in metric.compute().items():
                results[f"{name}/{key}"] = value
        return results

    def reset(self):
        for metric in self.metrics.values():
            metric.reset() 
            
def convert_batch_images_to_metric_format(pred_boxes: tf.Tensor, pred_scores: tf.Tensor, pred_labels: tf.Tensor, gt_boxes: tf.Tensor, gt_labels: tf.Tensor, gt_mask: tf.Tensor, image_ids: tf.Tensor):
    predictions = []
    ground_truths = []

    batch_size = tf.shape(pred_boxes)[0].numpy()

    for index in range(batch_size):
        image_id = image_ids[index].numpy()

        if isinstance(image_id, bytes):
            image_id = image_id.decode("utf-8")


        predictions.append({"image_id": image_id, "boxes": pred_boxes[index].numpy(), "scores": pred_scores[index].numpy(), "labels": pred_labels[index].numpy()})

        valid_mask = gt_mask[index].numpy()
        ground_truths.append({"image_id": image_id, "boxes": gt_boxes[index].numpy()[valid_mask], "labels": gt_labels[index].numpy()[valid_mask]})

    return predictions, ground_truths 

def build_metrics_from_config(config: dict[str, Any]):
    num_classes = config['num_classes']
    metrics_config = config.get("eval", {}).get("metrics", {})

    metrics = {}
    
    for name, config in metrics_config.items():
        metric_type = config.get("type", "voc_ap")
        iou_thresholds = config.get("iou_thresholds", [0.5])

        style = "coco" if metric_type == "coco_map" else "voc"

        metrics[name] = MeanAveragePrecision(num_classes = num_classes, iou_thresholds = iou_thresholds, style = style)

    return MetricsCollection(metrics)

def _box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single box and an array of boxes.
    
    Args:
        box:   shape (4,)  [x1, y1, x2, y2]
        boxes: shape (N,4) [x1, y1, x2, y2] for each box

    Returns:
        ious: shape (N,) IoU between `box` and each of `boxes`
    """
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)

    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    # Intersection coords
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    # Areas
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Union
    union = area_box + area_boxes - inter
    union = np.maximum(union, 1e-6)  # avoid division by zero

    return inter / union
