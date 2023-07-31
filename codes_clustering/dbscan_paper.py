import math
from t_SNE import *
from _Cluster_Plot import *
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

input_dir = '../files/PCA/PCA(1-48)'
file = '2022-12.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)
firm_names = data.index

ms = int(math.log(len(data_array)))

nbrs = NearestNeighbors(n_neighbors=ms + 1).fit(data_array)
distances, indices = nbrs.kneighbors(data_array)

# 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
avg_distances = np.mean(distances[:, 1:], axis=1)

# 결과 출력
print("Average distances to the nearest MinPts neighbor points:")
print(avg_distances)


def perform_DBSCAN(data_array):
    ms = int(math.log(len(data_array)))

    nbrs = NearestNeighbors(n_neighbors=ms + 1).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
    avg_distances = np.mean(distances[:, 1:], axis=1)
    avg_distances = sum(avg_distances) / len(data_array)

    eps = 0.4 * avg_distances

    print(ms)
    print(eps)

    lab = True
    if lab:
        # 결과 출력
        print("Average distances to the nearest MinPts neighbor points:")
        print(avg_distances)

        labels = DBSCAN(min_samples=ms, eps=eps, metric='manhattan').fit(data_array).labels_
        dbscan = DBSCAN(min_samples=ms, eps=eps, metric='manhattan')

    return labels, dbscan


if __name__ == "__main__":
    labels, dbscan = perform_DBSCAN(data_array)

    # Get the unique cluster labels
    unique_labels = sorted(list(set(labels)))

    clust = [[] for _ in unique_labels]
    for i, cluster_label in enumerate(labels):
        clust[unique_labels.index(cluster_label)].append(data.index[i])

    # 3. Print and plot the clusters
    for i, firms in enumerate(clust):
        plot_clusters(unique_labels[i], firms, data.index, data_array)  # Use the imported function

    # t_SNE(data_array, dbscan)
