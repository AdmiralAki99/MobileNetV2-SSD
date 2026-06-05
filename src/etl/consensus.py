from dataclasses import dataclass
from .detectors import Detection
from typing import Any
from scipy.optimize import linear_sum_assignment
import numpy as np


@dataclass
class ConsensusAnnotation:
    box: np.ndarray
    class_id: int
    class_name: str
    votes: int
    consensus_score: float
    model_confidence: dict[str, Any]


class ConsensusEngine:
    def __init__(self, config: dict[str, Any]):
        self.iou_threshold = config.get("consensus", {}).get("iou_threshold", 0.5)
        self.min_votes = config.get("consensus", {}).get("min_votes", 2)

    def compute(self, detections_per_model: dict[str, list[Detection]]):
        flattened_list = [
            (model_name, detection)
            for model_name, detection_list in detections_per_model.items()
            for detection in detection_list
        ]

        N = len(flattened_list)
        parent = list(range(N))

        offsets = {}
        offset = 0

        # Mapping the different models to a global structure
        for name, detections in detections_per_model.items():
            offsets[name] = offset
            offset = offset + len(detections)

        model_names = list(detections_per_model.keys())

        for index_1 in range(len(model_names)):
            for index_2 in range(index_1 + 1, len(model_names)):
                # Getting the detection for each  of the model
                boxes_a = np.array([detection.box for detection in detections_per_model[model_names[index_1]]])
                boxes_b = np.array([detection.box for detection in detections_per_model[model_names[index_2]]])

                iou_matrix = self._compute_iou_matrix(boxes_a=boxes_a, boxes_b=boxes_b)
                row_index, col_index = linear_sum_assignment(-iou_matrix)
                for row, col in zip(row_index, col_index):
                    if iou_matrix[row][col] >= self.iou_threshold:
                        # It passed the threshold for Annotations across two models
                        global_a = offsets[model_names[index_1]] + row
                        global_b = offsets[model_names[index_2]] + col
                        self._union(parent, global_a, global_b)

        clusters = {}
        for i in range(N):
            root = self._find(parent, i)
            if root not in clusters:
                clusters[root] = []

            clusters[root].append(i)

        consensus_annotations = []
        for root, indices in clusters.items():
            cluster_model_name = {flattened_list[i][0] for i in indices}
            if len(cluster_model_name) < self.min_votes:
                continue

            cluster_detections = [flattened_list[i][1] for i in indices]
            confidences = np.array([detection.confidence for detection in cluster_detections])
            weights = confidences / confidences.sum()
            avg_box = sum(weight * detection.box for weight, detection in zip(weights, cluster_detections))

            majority_class = max(
                set(detection.class_id for detection in cluster_detections),
                key=lambda c: sum(1 for detection in cluster_detections if detection.class_id == c),
            )
            class_name = next(
                detection.class_name for detection in cluster_detections if detection.class_id == majority_class
            )
            model_confidences = {flattened_list[i][0]: flattened_list[i][1].confidence for i in indices}

            consensus_annotations.append(
                ConsensusAnnotation(
                    box=avg_box,
                    class_id=majority_class,
                    class_name=class_name,
                    votes=len(cluster_model_name),
                    consensus_score=float(np.mean(confidences)),
                    model_confidence=model_confidences,
                )
            )

        return consensus_annotations

    def _find(self, parent, index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(self, parent, i, j):
        parent[self._find(parent, i)] = self._find(parent, j)

    def _compute_iou_matrix(self, boxes_a: np.ndarray, boxes_b: np.ndarray):
        iou_matrix = np.zeros((len(boxes_a), len(boxes_b)))

        for i, box_a in enumerate(boxes_a):
            for j, box_b in enumerate(boxes_b):
                x1a, y1a, x2a, y2a = box_a
                x1b, y1b, x2b, y2b = box_b

                intersection_x1 = max(x1a, x1b)
                intersection_y1 = max(y1a, y1b)
                intersection_x2 = min(x2a, x2b)
                intersection_y2 = min(y2a, y2b)

                intersection = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
                area_a = (x2a - x1a) * (y2a - y1a)
                area_b = (x2b - x1b) * (y2b - y1b)
                union = area_a + area_b - intersection

                iou_matrix[i][j] = intersection / union if union > 0 else 0.0

        return iou_matrix
