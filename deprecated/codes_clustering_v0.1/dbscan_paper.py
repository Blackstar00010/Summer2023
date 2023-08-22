import math
from t_SNE import *
from deprecated.Trash._Cluster_Plot import *
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

input_dir = '../files/PCA/PCA(1-48)'
file = '1993-01.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)
firm_names = data.index


def perform_DBSCAN(data_array):
    ms = int(math.log(len(data_array)))

    # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
    nbrs = NearestNeighbors(n_neighbors=ms + 1).fit(data_array)
    distances, indices = nbrs.kneighbors(data_array)
    avg_distances = np.mean(distances[:, 1:], axis=1)

    # Sort the average distances in ascending order
    sorted_distances = np.sort(avg_distances)

    # Calculate the index for the alpha percentile (alpha)
    alpha_percentile_index = int(len(sorted_distances) * 0.9)

    eps = sorted_distances[alpha_percentile_index]

    print(ms)
    print(eps)

    cluster_labels = DBSCAN(min_samples=ms, eps=eps, metric='manhattan').fit(data_array).labels_

    # Get the unique cluster labels
    unique_labels = sorted(list(set(cluster_labels)))

    clust = [[] for _ in unique_labels]
    for i, cluster_label in enumerate(cluster_labels):
        clust[unique_labels.index(cluster_label)].append(data.index[i])

    return clust


if __name__ == "__main__":
    labels = perform_DBSCAN(data_array)

    # Get the unique cluster labels
    unique_labels = sorted(list(set(labels)))

    clust = [[] for _ in unique_labels]
    for i, cluster_label in enumerate(labels):
        clust[unique_labels.index(cluster_label)].append(data.index[i])

    # Print and plot the clusters
    for i, firms in enumerate(clust):
        plot_clusters(unique_labels[i], firms, data.index, data_array)  # Use the imported function

    t_SNE(data_array, labels)
