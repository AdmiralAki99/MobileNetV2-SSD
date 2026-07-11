from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class ClusterFit:
    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    
class ClusteringStrategy(ABC):
    @abstractmethod
    def fit(self, points:np.ndarray, params: dict):
        pass