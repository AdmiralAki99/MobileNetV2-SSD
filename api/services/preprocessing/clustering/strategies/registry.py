from .base import ClusteringStrategy
from .kmeans import KMeansClusterer

def create_clusterer(name: str):
    match name:
        case "kmeans":
            # KMeans clustering
            return KMeansClusterer()
        case _:
            raise ValueError(f"Unknown clustering algorithm: {name}")