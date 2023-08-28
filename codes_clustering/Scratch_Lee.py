import math

import pandas as pd

from PCA_and_ETC import *
from sklearn.cluster import *
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import *
from scipy.spatial.distance import pdist
from scipy.spatial import distance


class Clustering:
    def __init__(self, data: pd.DataFrame):
        self.test = None
        self.PCA_Data = data
        self.index = data.index

        self.K_Mean = []
        self.DBSCAN = []
        self.Agglomerative = []
        self.Gaussian = []
        self.OPTIC = []
        self.HDBSCAN = []
        self.meanshift = []
        self.BIRCH = []

        self.K_Mean_labels = []
        self.DBSCAN_labels = []
        self.Agglomerative_labels = []
        self.Gaussian_labels = []
        self.OPTIC_labels = []
        self.Agglomerative_labels = []
        self.HDBSCAN_labels = []
        self.meanshift_labels = []
        self.BIRCH_labels = []

        self.lab = []
        self.lab_labels = []

    def outliers(self, k_value: int, alpha: float = 0.5):
        """
        :param k_value: the number of clusters
        :param alpha: the rate at which outliers are filtered
        :return: 2D list
        """
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)
        # Exclude the first column (firm names) & Exclude MOM_1

        kmeans = BisectingKMeans(n_clusters=k_value, init='k-means++', n_init=10, max_iter=500,
                                 algorithm='elkan', bisecting_strategy='largest_cluster').fit(self.PCA_Data)

        cluster_labels = kmeans.labels_  # Label of each point(ndarray of shape)

        self.test = kmeans
        self.K_Mean_labels = cluster_labels

        # Calculate outlier
        distance = kmeans.fit_transform(self.PCA_Data)  # Distance btw K centroid about all each points
        main_distance = np.min(distance, axis=1)  # Distance btw own K centroid about all each points

        main_distance_clustering = [[] for _ in range(k_value)]

        for cluster_num in range(k_value):
            distance_clustering = distance[cluster_labels == cluster_num]
            for i in range(len(distance_clustering)):
                main_distance_clustering[cluster_num].append(distance_clustering[i][cluster_num])

        # max distance in own cluster
        for i, cluster in enumerate(main_distance_clustering):
            if not cluster:
                continue
            main_distance_clustering[i] = np.max(cluster)

        max_main_distance_clustering = main_distance_clustering

        clusters = [[] for _ in range(k_value)]  # Cluster별 distance 분류
        clusters_index = [[] for _ in range(k_value)]  # Cluster별 index 분류

        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(main_distance[i])
            clusters_index[cluster_num].append(self.index[i])

        outliers = [[] for _ in range(k_value)]  # Cluster별 outliers' distance 분류
        for i, cluster in enumerate(clusters):
            if max_main_distance_clustering[i] == 0:
                continue
            for j, distance in enumerate(cluster):  # distance = 자기가 속한 클러스터 내에서 중심과의 거리, cluster별로 계산해야 함.
                if distance / max_main_distance_clustering[i] >= alpha:
                    # distance / 소속 cluster 점들 중 중심과 가장 먼 점의 거리 비율이 alpha 이상이면 outlier 분류
                    outliers[i].append(distance)

        outliers_index = []  # Cluster별 outliers's index 분류
        for i, cluster in enumerate(clusters):
            if not outliers[i]:
                continue
            for k, outlier_dis in enumerate(outliers[i]):
                for j, distance in enumerate(cluster):
                    if outlier_dis == distance:
                        outliers_index.append(clusters_index[i][j])
                    else:
                        continue

        outliers_index = list(set(outliers_index))

        # a에 있는 값을 b에서 빼기
        for value in outliers_index:
            for row in clusters_index:
                if value in row:
                    row.remove(value)

        clusters_index = [sublist for sublist in clusters_index if sublist]  # 빈 리스트 제거

        # 1차원 리스트로 전환된 outlier를 cluster 맨앞에 저장.
        clusters_index.insert(0, outliers_index)

        return clusters_index

    def perform_kmeans(self, k_value: int, alpha: float = 0.5):
        """
        :param k_value: k value to be tested
        :param alpha: the rate at which outliers are filtered
        :return: 3D list
        """
        clusters_k = []
        n_sample = self.PCA_Data.shape[0]  # number of values in the file
        # Skip if the number of values are less than k
        if n_sample <= k_value:
            k = n_sample
        clusters = self.outliers(k_value, alpha)
        clusters_k.append(clusters)

        self.K_Mean = clusters_k
        return self.K_Mean