from _table_generate import *
from sklearn.cluster import KMeans

input_dir = '../files/PCA/PCA(1-48)'
file = '2016-01.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)
firm_names = data.index


def outliers(data_array, K):
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
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
        for j, distance in enumerate(cluster):
            if distance == 0 or distance / max(cluster) >= 0.85:
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
    return clust


x = outliers(data_array, 5)
print('함수이용')
print(x)
for i in range(len(x)):
    print(x[i])

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

<<<<<<< HEAD
outliers = []
outlier_cluster_indices = []

outliers = [[] for _ in range(len(unique_labels))]

=======
for i in range(len(clusters)):
    print('cluster_index before')
    print(len(clusters_index[i]))

outliers = [[] for _ in range(5)]  # Cluster별 outliers's distance 분류

for i, cluster in enumerate(clusters):
    for j, distance in enumerate(cluster):
        if distance == 0 or distance / max(cluster) >= 0.85:
            outliers[i].append(distance)

outliers_index = [[] for _ in range(5)]  # Cluster별 outliers's index 분류
for i, cluster_dis in enumerate(clusters):
    for j, outlier_dis in enumerate(outliers[i]):
        for k, firm in enumerate(cluster_dis):
            if outlier_dis == firm:
                outliers_index[i].append(clusters_index[i][k])
                clusters_index[i].remove(clusters_index[i][k])  # 해당 index clusters_index에서 삭제
            else:
                continue

for i in range(len(clusters)):
    print('cluster_index')
    print(clusters_index[i])
    print('outliers_index')
    print(outliers_index[i])

for i in range(len(clusters)):
    print('cluster_index after')
    print(len(clusters_index[i]))

clusters_index = [sublist for sublist in clusters_index if sublist]
print(clusters_index)
outliers_index = [item for sublist in outliers_index for item in sublist]
print(outliers_index)
print(cluster_distance_min[163])
print(firm_names[163])

clus = []
for i in range(len(clusters_index)):
    clus.append(outliers_index)
    clus.append(clusters_index[i])
print('깡')
print(clus)
