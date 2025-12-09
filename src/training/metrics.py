from typing import List, Dict, Any
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class BaseMetric(ABC):
    def __init__(self, name:str):
        self._name = name
        
    @abstractmethod
    def update(self,preds: list[Dict[str,Any]], ground_truth: list[Dict[str,Any]]):
        pass
        
    @abstractmethod
    def compute(self):
        pass

    def reset(self):
        pass
        
    @property
    def name(self):
        return self._name
    
class VOCMAP(BaseMetric):
    def __init__(self, iou_thresh,num_classes: int,name: str):
        super().__init__(name = name)
        
        if isinstance(iou_thresh, (list, tuple)):
            self.iou_thresh = [float(t) for t in iou_thresh]
        else:
            self.iou_thresh = [float(iou_thresh)]
            
        self.num_classes = num_classes
        # Initialize the pred & gt lists
        self._preds = []
        self._ground_truth = []

    def reset(self):
        self._preds = []
        self._ground_truth = []

    def update(self,preds,ground_truth):
        for pred in preds:
            self._preds.append(
                (pred['image_id'], pred['boxes'], pred['scores'], pred['labels'])
            )

        for gt in ground_truth:
            self._ground_truth.append(
                (gt['image_id'], gt['boxes'], gt['labels'])
            )

    def compute(self):
        if len(self._ground_truth) == 0:
            results = {}
            for t in self.iou_thresh:
                results[f"mAP@{t}"] = 0.0
            return results

        combined = {}
        for t in self.iou_thresh:
            stats_t = self._compute_for_single_iou(t)
            combined.update(stats_t)

        return combined

    def _compute_for_single_iou(self, iou_thr):
    
        # GT Structures per class
        ground_truth_per_class = {c: {} for c in range(self.num_classes)}
        num_pos_per_class = {c: 0 for c in range(self.num_classes)}

        for image_id, gt_boxes, gt_labels in self._ground_truth:
            # Copying the boxes for calculations
            gt_boxes = np.asarray(gt_boxes,dtype = np.float32)
            gt_labels = np.asarray(gt_labels,dtype = np.int32)

            # Iterating over the boxes and their corresponding labels
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                class_num = int(gt_label)
                if class_num == 0:
                    # Background which is not needed
                    continue

                if image_id not in ground_truth_per_class[class_num]:
                    # Initial creation of the images records
                    ground_truth_per_class[class_num][image_id] = {
                        "boxes": [],
                        "detected": []
                    }

                ground_truth_per_class[class_num][image_id]["boxes"].append(gt_box)
                ground_truth_per_class[class_num][image_id]["detected"].append(False)
                num_pos_per_class[class_num] = num_pos_per_class[class_num] + 1

        # Making them into numpy arrays for easier reduction
        for class_num in range(self.num_classes):
            for image_id, data in ground_truth_per_class[class_num].items():
                data['boxes'] = np.asarray(data['boxes'], dtype=np.float32)
                data['detected'] = np.asarray(data["detected"], dtype=bool)

        # Calculating the predictions per class
        pred_per_class = {c: [] for c in range(self.num_classes)}

        for image_id, pred_box, pred_scores, pred_labels in self._preds:
            pred_box = np.asarray(pred_box,dtype = np.float32)
            pred_scores = np.asarray(pred_scores,dtype = np.float32)
            pred_labels = np.asarray(pred_labels,dtype = np.int32)

            # Iterating through all the predictions
            for bbox, score, label in zip(pred_box, pred_scores, pred_labels):
                class_num = int(label)
                if class_num == 0:
                    # Background which is not needed
                    continue

                pred_per_class[class_num].append({"image_id": image_id, "box": bbox, "score": float(score)})


        # Now calculating AP per class after creation of the data
        ap_per_class = {}

        for class_num in range(1, self.num_classes):
            preds_for_class = pred_per_class[class_num]
            num_pos = num_pos_per_class[class_num]

            if num_pos == 0:
                # There was no ground truth box for this class
                continue

            if len(preds_for_class) == 0:
                ap_per_class[class_num] = 0.0
                continue

            # Sorting the predictions by score
            preds_for_class.sort(key = lambda data: data['score'],reverse = True)

            TP = np.zeros(len(preds_for_class), dtype = np.float32)
            FP = np.zeros(len(preds_for_class), dtype = np.float32)

            for index, pred in enumerate(preds_for_class):
                image_id = pred['image_id']
                bbox = np.asarray(pred['box'], dtype = np.float32)

                if image_id not in ground_truth_per_class[class_num]:
                    FP[index] = 1.0
                    continue

                ground_truth_data = ground_truth_per_class[class_num][image_id]
                ground_truth_boxes = ground_truth_data['boxes']
                detected = ground_truth_data['detected']

                iou_matrix = _box_iou_xyxy(bbox, ground_truth_boxes)
                max_iou_per_index = int(np.argmax(iou_matrix)) if iou_matrix.size > 0 else -1
                max_iou = iou_matrix[max_iou_per_index] if iou_matrix.size > 0 else 0.0

                if max_iou >= iou_thr and not detected[max_iou_per_index]:
                    TP[index] = 1.0
                    detected[max_iou_per_index] = True
                else:
                    FP[index] = 1.0

            TP_cum = np.cumsum(TP)
            FP_cum = np.cumsum(FP)

            # Calculating the recall and precision
            recall = TP_cum / float(num_pos)
            precision = TP_cum / np.maximum(TP_cum + FP_cum, 1e-6) # division by zero safe guard

            # calculate the AP
            ap = self._voc_ap(recall,precision)
            ap_per_class[class_num] = float(ap)

        valid_aps = [ap for c, ap in ap_per_class.items() if num_pos_per_class[c] > 0]

        if len(valid_aps) == 0.0:
            mAP = 0.0
        else:
            mAP = float(np.mean(valid_aps))

        results = {
            f"mAP@{iou_thr}": mAP,
        }

        for c, ap in ap_per_class.items():
            if num_pos_per_class[c] > 0:
                results[f"mAP@{iou_thr}/class_{c}"] = ap

        return results

    def _voc_ap(self, recall: np.ndarray, precision: np.ndarray):
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        idx = np.where(mrec[1:] != mrec[:-1])[0]

        ap = 0.0
        for i in idx:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]

        return ap

class COCOMAP(BaseMetric):
    def __init__(self, iou_thresh: list[float] | None,num_classes: int,name: str):
        super().__init__(name = name)
        self.num_classes = num_classes

        if iou_thresh is None:
            iou_thresh = [0.5 + 0.05 * i for i in range(10)]

        if isinstance(iou_thresh, (list, tuple)):
            self.iou_thresh = [float(t) for t in iou_thresh]
        else:
            self.iou_thresh = [float(iou_thresh)]
        
        # Initialize the pred & gt lists
        self._preds = []
        self._ground_truth = []

    def reset(self):
        self._preds = []
        self._ground_truth = []

    def update(self,preds,ground_truth):
        for pred in preds:
            self._preds.append(
                (pred['image_id'], pred['boxes'], pred['scores'], pred['labels'])
            )

        for gt in ground_truth:
            self._ground_truth.append(
                (gt['image_id'], gt['boxes'], gt['labels'])
            )

    def compute(self):
        if len(self._ground_truth) == 0:
            # No GT at all: define all metrics as 0
            key = f"{self.name}/mAP@[{self.iou_thresh[0]:.2f}:{self.iou_thresh[-1]:.2f}]"
            return {key: 0.0}

        num_iou = len(self.iou_thresh)

        # GT Structures per class
        ground_truth_per_class = {c: {} for c in range(self.num_classes)}
        num_pos_per_class = {c: 0 for c in range(self.num_classes)}

        for image_id, gt_boxes, gt_labels in self._ground_truth:
            # Copying the boxes for calculations
            gt_boxes = np.asarray(gt_boxes,dtype = np.float32)
            gt_labels = np.asarray(gt_labels,dtype = np.int32)

            # Iterating over the boxes and their corresponding labels
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                class_num = int(gt_label)
                if class_num == 0:
                    # Background which is not needed
                    continue

                if image_id not in ground_truth_per_class[class_num]:
                    # Initial creation of the images records
                    ground_truth_per_class[class_num][image_id] = {
                        "boxes": [],
                    }

                ground_truth_per_class[class_num][image_id]["boxes"].append(gt_box)
                num_pos_per_class[class_num] = num_pos_per_class[class_num] + 1

        for class_num in range(self.num_classes):
            for image_id, data in ground_truth_per_class[class_num].items():
                data['boxes'] = np.asarray(data['boxes'], dtype=np.float32)
                
        # Calculating the predictions per class
        pred_per_class = {c: [] for c in range(self.num_classes)}

        for image_id, pred_box, pred_scores, pred_labels in self._preds:
            pred_box = np.asarray(pred_box,dtype = np.float32)
            pred_scores = np.asarray(pred_scores,dtype = np.float32)
            pred_labels = np.asarray(pred_labels,dtype = np.int32)

            # Iterating through all the predictions
            for bbox, score, label in zip(pred_box, pred_scores, pred_labels):
                class_num = int(label)
                if class_num == 0:
                    # Background which is not needed
                    continue

                pred_per_class[class_num].append({"image_id": image_id, "box": bbox, "score": float(score)})

        AP = np.full((self.num_classes, num_iou), np.nan, dtype=np.float32)

        for class_num in range(1, self.num_classes):
            preds_for_class = pred_per_class[class_num]
            num_pos = num_pos_per_class[class_num]

            if num_pos == 0:
                # There was no ground truth box for this class
                continue

            if len(preds_for_class) == 0:
                AP[class_num] = 0.0
                continue

            # Sorting the predictions by score
            preds_for_class.sort(key = lambda data: data['score'],reverse = True)

            for index, iou_thr in enumerate(self.iou_thresh):
                detected_flags = {}
                
                for image_id, data in ground_truth_per_class[class_num].items():
                    num_gt = data["boxes"].shape[0]
                    detected_flags[image_id] = np.zeros(num_gt, dtype=bool)

                TP = np.zeros(len(preds_for_class), dtype = np.float32)
                FP = np.zeros(len(preds_for_class), dtype = np.float32)

                for pred_index, pred in enumerate(preds_for_class):
                    image_id = pred['image_id']
                    bbox = np.asarray(pred['box'], dtype = np.float32)

                    if image_id not in ground_truth_per_class[class_num]:
                        FP[pred_index] = 1.0
                        continue

                    ground_truth_data = ground_truth_per_class[class_num][image_id]
                    ground_truth_boxes = ground_truth_data['boxes']
                    det_flags = detected_flags[image_id]

                    iou_matrix = _box_iou_xyxy(bbox, ground_truth_boxes)
                    if iou_matrix.size == 0:
                        FP[pred_index] = 1.0
                        continue

                    max_iou_index = int(np.argmax(iou_matrix))
                    max_iou = float(iou_matrix[max_iou_index])

                    if max_iou >= iou_thr and not det_flags[max_iou_index]:
                        TP[pred_index] = 1.0
                        det_flags[max_iou_index] = True
                    else:
                        FP[pred_index] = 1.0

                TP_cum = np.cumsum(TP)
                FP_cum = np.cumsum(FP)

                recall = TP_cum / float(num_pos)
                precision = TP_cum / np.maximum(TP_cum + FP_cum, 1e-6) # division by zero safe guard

                AP[class_num,index] = self._coco_ap_101(recall, precision)
                
        valid_classes  = [c for c in range(1, self.num_classes) if num_pos_per_class[c] > 0]

        if len(valid_classes) == 0:
            mAP = 0.0
            ap50 = 0.0
            ap75 = 0.0
        else:
            AP_valid = AP[valid_classes, :]
            mAP = float(np.nanmean(AP_valid))
            
            ap50 = None
            ap75 = None

            if 0.5 in self.iou_thresh:
                t50_idx = self.iou_thresh.index(0.5)
                ap50 = float(np.nanmean(AP_valid[:, t50_idx]))
            if 0.75 in self.iou_thresh:
                t75_idx = self.iou_thresh.index(0.75)
                ap75 = float(np.nanmean(AP_valid[:, t75_idx]))

        key_main = f"{self.name}/mAP@[{self.iou_thresh[0]:.2f}:{self.iou_thresh[-1]:.2f}]"
        results: dict[str, float] = {key_main: mAP}

        if ap50 is not None:
            results[f"{self.name}/AP@0.50"] = ap50
        if ap75 is not None:
            results[f"{self.name}/AP@0.75"] = ap75

        return results

    def _coco_ap_101(self, recall: np.ndarray, precision: np.ndarray):
        
        if recall.size == 0:
            return 0.0

        rec = np.asarray(recall, dtype=np.float32)
        prec = np.asarray(precision, dtype=np.float32)

        recall_samples = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        precisions_interp = np.zeros_like(recall_samples)

        for i, r in enumerate(recall_samples):
            # precision at recall >= r
            mask = rec >= r
            if np.any(mask):
                precisions_interp[i] = np.max(prec[mask])
            else:
                precisions_interp[i] = 0.0

        return float(np.mean(precisions_interp))
  
class MetricsManager:
    def __init__(self, metrics:list[BaseMetric], prefix: str | None = None):
        self.metrics = metrics
        self.prefix = prefix or ""

    def update(self, pred: dict[str,Any], ground_truth: dict[str,Any]):
        for metric in self.metrics:
            metric.update(pred,ground_truth)

    def compute(self):
        combined: dict[str,float] = {}
        for metric in self.metrics:
            stat = metric.compute()
            for key,value in stat.items():
                k = f"{self.prefix}{metric.name}/{key}"
                combined[k] = value
        return combined
        
    def reset(self):
        for metric in self.metrics:
            metric.reset()
            
def build_metrics_config(config: dict):

    eval_cfg = config["eval"]
    metric_cfg = eval_cfg["metrics"]
    num_classes = config["model"]["num_classes"]
    
    metrics: list[BaseMetric] = []

    for metric_name, mc in metric_cfg.items():
        metric_type = mc.get("type", "voc_ap")
        if metric_type == "voc_ap":
            metrics.append(
                VOCMAP(
                    iou_thresh=mc.get("iou_thresholds", [0.5]),
                    num_classes=num_classes,
                    name=metric_name,  # use key from YAML
                )
            )

        elif metric_type == "coco_map":
            metrics.append(
                COCOMAP(
                    iou_thresh=mc.get("iou_thresholds",
                                          [0.5, 0.55, 0.6, 0.65,
                                           0.7, 0.75, 0.8, 0.85,
                                           0.9, 0.95]),
                    num_classes=num_classes,
                    name=metric_name,
                )
            )

        else:
            raise ValueError(f"Unknown metric type: {metric_type!r} for {metric_name!r}")

    return MetricsManager(metrics=metrics, prefix=f"{eval_cfg['dataset_split']}/")

def _box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:

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

    