from _table_generate import *
from sklearn.cluster import KMeans

input_dir = '../files/PCA/PCA(1-48)'
file = '2016-01.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)  # Exclude the first column (firm names) & Exclude MOM_1
firm_names = data.index  # Get the first column (firm names)

kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
X = kmeans.fit(data_array)
cluster_labels = kmeans.labels_
print('각 회사가 속한 클러스터의 라벨')
print(cluster_labels)

distance = kmeans.fit_transform(data_array)
# 중심 5개와 각 포인터들 사이의 거리
print('각 회사가 속한 클러스터에서 중심까지 거리')
# 최소거리의 의미는 자기가 속한 클러스터 내에서의 중심과 거리를 의미
cluster_distance_min = np.min(distance, axis=1)
print(cluster_distance_min)

unique_labels = sorted(list(set(cluster_labels)))

clusters = [[] for _ in range(len(unique_labels))]
clusters_index = [[] for _ in range(len(unique_labels))]
for i, cluster_num in enumerate(cluster_labels):
    clusters[cluster_num].append(cluster_distance_min[i])
    clusters_index[cluster_num].append(data.index[i])

outliers = []
outlier_cluster_indices = []

outliers = [[] for _ in range(len(unique_labels))]

for i, cluster in enumerate(clusters):
    for j, distance in enumerate(cluster):
        if distance == 0 or distance / max(cluster) >= 0.85:
            outliers.append(distance)
            outlier_cluster_indices.append(i)

print('outlier의 거리')
print(outliers)
print(outlier_cluster_indices)

print('distance를 클러스터로 나눔')
print(clusters)

for i in range(len(clusters)):
    print(clusters[i])
    print(clusters_index[i])

# for i in range(outliers):
#     for j in range(0, len(clusters)):
#         if outliers[i] == j + 1:
#             clusters[j + 1] = 0

# filtered_distances = [distance for distance in cluster if distance / max(cluster) <= 0.9]
# outliers.append(filtered_distances)
