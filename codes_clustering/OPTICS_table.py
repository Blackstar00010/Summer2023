from _table_generate import *
from sklearn.cluster import OPTICS

# 파일 불러오기
input_dir = '../files/Clustering/PCA(1-48)'
output_dir = '../files/Clustering/OPTICS'
optics = sorted(filename for filename in os.listdir(input_dir))

# CSV 파일 하나에 대해서 실행
for file in optics:
    data = read_and_preprocess_data(input_dir, file)
    mat = data.values[:, 1:]

    # 1. OPTICS 알고리즘

    if len(data) < 3:
        min_samples = int(len(data))
        min_cluster_size = int(len(data))

    else:
        min_samples = 3
        min_cluster_size = 3

    xi = 0.1

    clust = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    # xi = 거리, min_samples = 포함할 최소 데이터 수, min_cluster_size는 생성될 최소 군집 수

    cluster_labels = clust.fit_predict(mat)

    unique_labels = list(set(cluster_labels))

    clusters = [[] for _ in range(len(unique_labels))]

    for i, cluster in enumerate(cluster_labels):
        clusters[unique_labels.index(cluster)].append(data.index[i])

    new_table_generate(data, clusters, output_dir, file)
