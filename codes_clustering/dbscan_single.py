from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from t_SNE import *
from _Cluster_Plot import *

input_dir = '../files/PCA/PCA(1-48)'
file = '1993-01.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)
firm_names = data.index  # Get the first column (firm names)

# Define DBSCAN parameters
eps = 2.404  # Maximum distance between two samples to be considered as neighbors
min_samples = 2  # Minimum number of samples in a neighborhood for a point to be considered as a core point


def perform_DBSCAN(data_array, eps, min_samples):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
    cluster_labels = dbscan.fit_predict(data_array)

    return cluster_labels, dbscan


cluster_labels, dbscan = perform_DBSCAN(data_array, eps, min_samples)

# Get the unique cluster labels
unique_labels = sorted(list(set(cluster_labels)))

clust = [[] for _ in unique_labels]
for i, cluster_label in enumerate(cluster_labels):
    clust[unique_labels.index(cluster_label)].append(data.index[i])

# 3. Print and plot the clusters
for i, firms in enumerate(clust):
    plot_clusters(unique_labels[i], firms, data.index, data_array)  # Use the imported function

t_SNE(data_array, dbscan)

lab = True
if lab:
    # NearestNeighbors 모델 생성
    nn_model = NearestNeighbors(n_neighbors=20, metric='manhattan')
    nn_model.fit(data_array)

    # 각 데이터 포인트에 대한 최근접 이웃 인덱스와 거리 계산
    distances, indices = nn_model.kneighbors(data_array)
    print(distances)
    print(indices)

    # 각 데이터 포인트의 평균 최근접 이웃 거리 계산
    average_distances = np.mean(distances[:, 1:], axis=1)

    # 클러스터 레이블 출력 및 평균 최근접 이웃 거리 출력
    print("DBSCAN Cluster Labels:", cluster_labels)
    print("Average Distances to MinPts Neighbors:", average_distances)

    average_distance = sum(average_distances) / len(average_distances)
    print(average_distance * 0.5)
