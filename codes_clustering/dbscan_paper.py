from sklearn.neighbors import NearestNeighbors
import numpy as np

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        neighbors = NearestNeighbors(radius=self.eps).fit(X)
        neighborhoods = neighbors.radius_neighbors(X, return_distance=False)

        core_samples = np.array([len(neighbors) >= self.min_samples for neighbors in neighborhoods])
        labels = np.full(X.shape[0], -1)
        cluster_id = 0

        for point in range(X.shape[0]):
            if not core_samples[point] or labels[point] != -1:
                continue

            labels[point] = cluster_id
            self._expand_cluster(X, labels, core_samples, neighborhoods, point, cluster_id)
            cluster_id += 1

        self.labels_ = labels
        return self

    def _expand_cluster(self, X, labels, core_samples, neighborhoods, point, cluster_id):
        for neighbor in neighborhoods[point]:
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
                if core_samples[neighbor]:
                    self._expand_cluster(X, labels, core_samples, neighborhoods, neighbor, cluster_id)

    def predict(self, X):
        return self.labels_
