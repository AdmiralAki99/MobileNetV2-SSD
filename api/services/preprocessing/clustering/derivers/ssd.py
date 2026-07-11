from .base import PriorsDeriver
import numpy as np
from ..results import ClusterResultSSD
from ..strategies.base import ClusteringStrategy

class SSDPriorDeriver(PriorsDeriver):
    def derive_priors(self, boxes: np.ndarray, strategy: ClusteringStrategy, params: dict):
        box_dimensions = boxes['norm']
        W, H = box_dimensions[:,0], box_dimensions[:,1]
        scale = np.sqrt(W*H)
        aspect_ratio = W/H
        
        fit = strategy.fit(points=aspect_ratio.reshape(-1,1), params={'k': params['num_aspect_ratios']})
        aspect_ratios = sorted(round(float(centroid), 4) for centroid in fit.centroids.flatten())
        
        min_scale = float(np.percentile(scale, params.get("scale_low_pct", 2)))
        max_scale = float(np.percentile(scale, params.get("scale_high_pct", 98)))
        
        fitness = self._fitness(box_dimensions, min_scale, max_scale, aspect_ratios, params)
        return ClusterResultSSD(
            centroids=fit.centroids,
            fitness=fitness,
            min_scale=min_scale,
            max_scale=max_scale,
            aspect_ratios=aspect_ratios,
            dataset=params['dataset']
        )
        
    def _fitness(self, box_dims, min_scale, max_scale, aspect_ratios, params):
        n = params["num_levels"]
        scales = np.linspace(min_scale, max_scale, n)
        anchors = np.array([[scale*np.sqrt(ratio), scale/np.sqrt(ratio)] for scale in scales for ratio in aspect_ratios])
        iou = self._wh_iou(box_dims, anchors)
        best_iou = iou.max(axis=1)
        
        return {
            "mean_iou": float(best_iou.mean()),
            "recall@0.5":float((best_iou > 0.5).mean()) 
        }
        
    @staticmethod
    def _wh_iou(boxes, anchors):
        bw, bh = boxes[:, None, 0], boxes[:, None, 1]
        aw, ah = anchors[None, :, 0], anchors[None, :, 1]
        inter = np.minimum(bw, aw) * np.minimum(bh, ah)
        return inter / (bw*bh + aw*ah - inter + 1e-9)