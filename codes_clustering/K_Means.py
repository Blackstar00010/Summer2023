from _table_generate import *
from sklearn.cluster import KMeans
from _Cluster_Plot import plot_clusters
import numpy as np

# Clusters the firms using K-Means algorithm
# Performs just one CSV file
# Read data from CSV file
input_dir = '../files/PCA/PCA(1-48)'
file = '2016-01.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)  # Exclude the first column (firm names) & Exclude MOM_1
firm_names = data.index  # Get the first column (firm names)

# Define the number of clusters k
k_values = [5, 10]


def perform_kmeans(k_values, data_array, firm_names):
    # Perform k-means clustering for each value of k
    clusters_k = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # n_init setting to suppress warning
        kmeans.fit(data_array)  # Compute K-Means clustering
        cluster_labels = kmeans.labels_  # Label of each point (ndarray of shape)

        clusters = [[] for _ in range(k)]

        for i, cluster in enumerate(cluster_labels):
            # i: firm index
            # cluster: cluster index
            clusters[cluster].append(firm_names[i])

        clusters_k.append(clusters)
    return clusters_k


def outliers(k_values, data_array, firm_names):
    # Perform K-Means clustering for each k
    clusters_k = perform_kmeans(k_values, data_array, firm_names)

    # Identify outliers in each cluster for each k
    outliers_k = []
    for clusters in clusters_k:
        outliers = []
        for cluster in clusters:
            # Get the data for the firms in the cluster
            cluster_data = data_array[np.isin(firm_names, cluster)]

            # Perform K-Means clustering on the cluster data
            kmeans = KMeans(n_clusters=1, n_init=10, random_state=0)
            kmeans.fit(cluster_data)

            # Calculate distances to the cluster center
            distances = kmeans.transform(cluster_data)
            min_distances = np.min(distances, axis=1)

            # Identify outliers as points whose distance from the cluster center
            # is 85% or more of the maximum distance within the cluster
            cluster_outliers = firm_names[np.isin(firm_names, cluster)][min_distances / min_distances.max() >= 0.85]
            outliers.append(list(cluster_outliers))

        outliers_k.append(outliers)

    return clusters_k, outliers_k


if __name__ == "__main__":
    clusters_k, outliers_k = outliers(k_values, data_array, firm_names)

    for k, (clusters, outliers) in enumerate(zip(clusters_k, outliers_k)):
        print(f"For k = {k_values[k]}:")
        for i, (cluster, outlier) in enumerate(zip(clusters, outliers)):
            print(f"  Cluster {i+1}: {cluster}")
            print(f"  Outliers: {outlier}")
            plot_clusters(i, outlier, firm_names, data_array)



'''def outliers(data_array, K):
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    kmeans.fit(data_array)
    cluster_labels = kmeans.labels_  # 각 회사가 속한 Cluster Label
    distance = kmeans.fit_transform(data_array)  # K개의 중심점과 각 포인트들 사이의 거리
    cluster_distance_min = np.min(distance, axis=1)  # 자기가 속한 클러스터 내에서 중심과의 거리

    clusters = [[] for _ in range(K)]  # Cluster별 distance 분류
    clusters_index = [[] for _ in range(K)]  # Cluster별 index 분류
    for i, cluster_num in enumerate(cluster_labels):
        clusters[cluster_num].append(cluster_distance_min[i])
        clusters_index[cluster_num].append(data.index[i])

    outliers = [[] for _ in range(K)]  # Cluster별 outliers's distance 분류
    for i, cluster in enumerate(clusters):
        for j, distance in enumerate(cluster):  # distance = 자기가 속한 클러스터 내에서 중심과의 거리, cluster별로 계산해야 함.
            if distance == 0 or distance / max(cluster) >= 0.85:  # distance / 소속 cluster 점들 중 중심과 가장 먼 점의 거리 비율이 85%이상이면 outlier 분류
                outliers[i].append(distance)

    outliers_index = [[] for _ in range(K)]  # Cluster별 outliers's index 분류
    for i, cluster_dis in enumerate(clusters):
        for j, outlier_dis in enumerate(outliers[i]):
            for k, firm in enumerate(cluster_dis):
                if outlier_dis == firm:
                    outliers_index[i].append(clusters_index[i][k])
                    clusters_index[i].remove(clusters_index[i][k])  # 해당 index clusters_index에서 삭제
                else:
                    continue

    clusters_index = [sublist for sublist in clusters_index if sublist]  # 빈 리스트 제거
    outliers_index = [item for sublist in outliers_index for item in sublist]  # 2차원 리스트 1차원으로

    clust = []
    clust.append(outliers_index)
    for i in range(len(clusters_index)):
        clust.append(clusters_index[i])

    return clust'''



'''
#Define the number of clusters k
k_values = [5, 10]


def perform_kmeans(k_values, data_array, firm_names):
    # Perform k-means clustering for each value of k
    clusters_k = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # n_init setting to suppress warning
        kmeans.fit(data_array)  # Compute K-Means clustering
        cluster_labels = kmeans.labels_  # Label of each point (ndarray of shape)

        clusters = [[] for _ in range(k)]

        for i, cluster in enumerate(cluster_labels):
            # i: firm index
            # cluster: cluster index
            clusters[cluster].append(firm_names[i])

        clusters_k.append(clusters)
    return clusters_k

if __name__ == "__main__":
    clusters_k = perform_kmeans(k_values, data_array, firm_names)

    print(clusters_k)
    # Print the clusters for each k value & plot the clusters
    for i, clusters in enumerate(clusters_k):
        print(f'Clusters for k = {k_values[i]}:')
        for j, firms in enumerate(clusters):
            plot_clusters(j, firms, firm_names, data_array)  # Use the imported function
'''