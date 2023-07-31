from t_SNE import *
from sklearn.cluster import KMeans
from _Cluster_Plot import plot_clusters

# Clusters the firms using K-Means algorithm
# Performs just one CSV file
# Read data from CSV file
input_dir = '../files/PCA/PCA(1-48)_adj'
file = '2017-01.csv'
# file = '1992-09.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)  # Exclude the first column (firm names) & Exclude MOM_1
firm_names = data.index  # Get the first column (firm names)


def outliers(data_array, firm_names, K):
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    kmeans.fit(data_array)
    cluster_labels = kmeans.labels_  # Label of each point(ndarray of shape)
    distance = kmeans.fit_transform(data_array)  # Distance btw K central points and each point
    cluster_distance_min = np.min(distance, axis=1)  # Distance between the point and the central point

    clusters = [[] for _ in range(K)]  # Cluster별 distance 분류

    clusters_index = [[] for _ in range(K)]  # Cluster별 index 분류
    for i, cluster_num in enumerate(cluster_labels):
        clusters[cluster_num].append(cluster_distance_min[i])
        clusters_index[cluster_num].append(firm_names[i])

    outliers = [[] for _ in range(K)]  # Cluster별 outliers' distance 분류
    for i, cluster in enumerate(clusters):
        for j, distance in enumerate(cluster):  # distance = 자기가 속한 클러스터 내에서 중심과의 거리, cluster별로 계산해야 함.
            if distance == 0 or distance / max(
                    cluster) >= 0.5:  # distance / 소속 cluster 점들 중 중심과 가장 먼 점의 거리 비율이 85%이상이면 outlier 분류
                outliers[i].append(distance)

    outliers_index = []  # Cluster별 outliers's index 분류
    for i, cluster_dis in enumerate(clusters):
        for j, outlier_dis in enumerate(outliers[i]):
            for k, firm in enumerate(cluster_dis):
                if outlier_dis == firm:
                    outliers_index.append(clusters_index[i][k])
                else:
                    continue

    # a에 있는 값을 b에서 빼기
    for value in outliers_index:
        for row in clusters_index:
            if value in row:
                row.remove(value)

    clusters_index = [sublist for sublist in clusters_index if sublist]  # 빈 리스트 제거

    clust = []
    clust.append(outliers_index)
    for i in range(len(clusters_index)):
        clust.append(clusters_index[i])

    return clust


def perform_kmeans(k_values, data_array, firm_names):
    clusters_k = []
    for k in k_values:
        clust = outliers(data_array, firm_names, k)
        clusters_k.append(clust)
    return clusters_k


if __name__ == "__main__":
    # Define the number of clusters k
    k_values = [5, 10]

    clusters_k = perform_kmeans(k_values, data_array, firm_names)
    # Print the clusters for each k value & plot the clusters
    for i, clusters in enumerate(clusters_k):
        print(f'Clusters for k = {k_values[i]}:')
        for j, firms in enumerate(clusters):
            plot_clusters(j - 1, firms, firm_names, data_array)  # Use the imported function

'''first = False
if first:
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
    kmeans.fit(data_array)
    cluster_labels = kmeans.labels_  # 각 회사가 속한 Cluster Label
    distance = kmeans.fit_transform(data_array)  # K개의 중심점과 각 포인트들 사이의 거리
    cluster_distance_min = np.min(distance, axis=1)  # 자기가 속한 클러스터 내에서 중심과의 거리

    clusters = [[] for _ in range(5)]  # Cluster별 distance 분류
    clusters_index = [[] for _ in range(5)]  # Cluster별 index 분류
    for i, cluster_num in enumerate(cluster_labels):
        clusters[cluster_num].append(cluster_distance_min[i])
        clusters_index[cluster_num].append(data.index[i])

    outliers = [[] for _ in range(5)]  # Cluster별 outliers's distance 분류
    for i, cluster in enumerate(clusters):
        for j, distance in enumerate(cluster):  # distance = 자기가 속한 클러스터 내에서 중심과의 거리, cluster별로 계산해야 함.
            if distance == 0 or distance / max(
                    cluster) >= 0.85:  # distance / 소속 cluster 점들 중 중심과 가장 먼 점의 거리 비율이 85%이상이면 outlier 분류
                outliers[i].append(distance)

    outliers_index = [[] for _ in range(5)]  # Cluster별 outliers's index 분류
    for i, cluster_dis in enumerate(clusters):
        for j, outlier_dis in enumerate(outliers[i]):
            for k, firm in enumerate(cluster_dis):
                if outlier_dis == firm:
                    outliers_index[i].append(clusters_index[i][k])
                else:
                    continue

    outliers_index = [item for sublist in outliers_index for item in sublist]  # 2차원 리스트 1차원으로

    # a에 있는 값을 b에서 빼기
    for value in outliers_index:
        for row in clusters_index:
            if value in row:
                row.remove(value)

    clusters_index = [sublist for sublist in clusters_index if sublist]  # 빈 리스트 제거

    clust = []
    clust.append(outliers_index)
    for i in range(len(clusters_index)):
        clust.append(clusters_index[i])

    for i in range(len(clust)):
        print(clust[i])'''
