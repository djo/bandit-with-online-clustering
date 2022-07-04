import numpy as np


class SequentialKMeans:
    def __init__(self, init_cluster_centers: np.ndarray, gamma: float):
        assert 0.0 <= gamma <= 1.0
        self.gamma = gamma
        self.num_clusters = len(init_cluster_centers)
        self.cluster_centers = np.copy(init_cluster_centers)
        self.m = np.copy(self.cluster_centers)
        self.w = np.ones(self.num_clusters, dtype=float)

    def fit(self, x) -> int:
        """
        Fit a new data point x

        :param x: data point
        :return: cluster the data point assigned to
        """
        distances = np.linalg.norm(x - self.cluster_centers, axis=1)
        i = np.argmin(distances)
        self.m[i] = self.m[i] * self.gamma + x
        self.w[i] = self.w[i] * self.gamma + 1
        self.cluster_centers[i] = self.m[i] / self.w[i]
        return int(i)
