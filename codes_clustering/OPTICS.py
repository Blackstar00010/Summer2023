from _table_generate import *
from _Cluster_Plot import plot_clusters
from sklearn.cluster import OPTICS

# 파일 불러오기
input_dir = '../files/PCA/PCA(1-48)'
file = '2018-01.csv'
data = read_and_preprocess_data(input_dir, file)
mat = data.values[:, 1:].astype(float)

# 1. OPTICS
clust = OPTICS(min_samples=3, xi=0.1, min_cluster_size=3)
# min_samples = 포함할 최소 데이터 수, xi = 거리, min_cluster_size = 생성될 최소 군집 수

cluster_labels = clust.fit_predict(mat)

unique_labels = list(set(cluster_labels))

clusters = [[] for _ in range(len(unique_labels))]

for i, cluster in enumerate(cluster_labels):
    clusters[unique_labels.index(cluster)].append(data.index[i])

# 2. Print and plot the clusters
for i, firms in enumerate(clusters):
    plot_clusters(unique_labels[i], firms, data.index, mat)
