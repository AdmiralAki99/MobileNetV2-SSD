import numpy as np
from sklearn.cluster import KMeans
from .base import ClusteringStrategy, ClusterFit

# Defining the Strategy for the KMeans Clustering
class KMeansClusterer(ClusteringStrategy):
    def fit(self, points, params):
        model = KMeans(n_clusters=params['k'], random_state=params.get("seed",42), n_init=params.get("n_init",10))
        labels = model.fit_predict(points)
        return ClusterFit(
            centroids=model.cluster_centers_,
            labels=labels,
            inertia=float(model.inertia_)
        )