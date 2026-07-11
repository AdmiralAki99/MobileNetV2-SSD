from dataclasses import dataclass
import numpy as np

@dataclass
class ClusterResult:
    centroids: np.ndarray
    fitness: dict
    dataset: str
    
@dataclass
class ClusterResultSSD(ClusterResult):
    min_scale: float
    max_scale: float
    aspect_ratios: list[float]
    
    def to_dict(self):
        return {
            "dataset": self.dataset,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "aspect_ratios": self.aspect_ratios,
            "fitness": self.fitness,
            "centroids": self.centroids.tolist(),   # ndarray → list for JSON
        } 