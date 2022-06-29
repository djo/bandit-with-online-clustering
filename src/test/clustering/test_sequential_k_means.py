import numpy as np
import pytest
from numpy.random import Generator, PCG64
from numpy.testing import assert_array_almost_equal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, pairwise_distances_argmin

from src.clustering.sequential_k_means import SequentialKMeans


class TestSequentialKMeans:
    def test_fit(self):
        km = SequentialKMeans(np.array([[0.2, 0.2], [0.8, 0.8]]), gamma=1.0)

        assert km.fit(np.array([0.2, 0.2])) == 0
        assert km.fit(np.array([0.8, 0.8])) == 1
        assert_array_almost_equal(km.cluster_centers, [[0.2, 0.2], [0.8, 0.8]])

        assert km.fit(np.array([0.5, 0.5])) == 0
        assert_array_almost_equal(km.cluster_centers, [[0.3, 0.3], [0.8, 0.8]])

    def test_fit_with_discounting_factor(self):
        km = SequentialKMeans(np.array([[0.2, 0.2], [0.8, 0.8]]), gamma=0.9)

        assert km.fit(np.array([0.2, 0.2])) == 0
        assert km.fit(np.array([0.8, 0.8])) == 1
        assert_array_almost_equal(km.cluster_centers, [[0.2, 0.2], [0.8, 0.8]])

        assert km.fit(np.array([0.5, 0.5])) == 0
        assert_array_almost_equal(km.cluster_centers, [[0.3107, 0.3107], [0.8, 0.8]])

    def test_clustering_quality(self):
        random = Generator(PCG64(41))
        data_points, true_clusters = make_blobs(
            n_samples=300, n_features=2, centers=3, cluster_std=2.5, shuffle=False, random_state=42
        )

        idx = random.permutation(data_points.shape[0])[:3]
        init_centers = np.copy(data_points[idx])

        skm = SequentialKMeans(init_centers, gamma=0.95)
        online_clusters = np.full(len(data_points), fill_value=-1, dtype=int)
        for i, x in enumerate(data_points):
            online_clusters[i] = skm.fit(x)

        assert adjusted_rand_score(true_clusters, online_clusters) == pytest.approx(0.71, rel=1e-2)

        skm_clusters = pairwise_distances_argmin(data_points, skm.cluster_centers)
        assert adjusted_rand_score(true_clusters, skm_clusters) == pytest.approx(0.93, rel=1e-2)

        km = KMeans(n_clusters=3, n_init=1, init=init_centers, random_state=random.integers(100))
        km_clusters = km.fit_predict(data_points)
        assert adjusted_rand_score(true_clusters, km_clusters) == pytest.approx(0.94, rel=1e-2)
