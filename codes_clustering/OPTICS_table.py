from _table_generate import *
from sklearn.cluster import OPTICS

# 파일 불러오기
input_dir = '../files/PCA/PCA(1-48)_adj'
output_dir = '../files/Clustering_adj/OPTICS'
optics = sorted(filename for filename in os.listdir(input_dir))

# CSV 파일 하나에 대해서 실행
for file in optics:
    data = read_and_preprocess_data(input_dir, file)
    data_array = data.values[:, 1:]

    cluster_labels = OPTICS(cluster_method='xi', metric='euclidean').fit(data_array).labels_
    # xi = 거리, min_samples = 포함할 최소 데이터 수, min_cluster_size는 생성될 최소 군집 수

    unique_labels = list(set(cluster_labels))

    clusters = [[] for _ in range(len(unique_labels))]

    for i, cluster in enumerate(cluster_labels):
        clusters[unique_labels.index(cluster)].append(data.index[i])

    # 2. Save CSV
    # columns = ['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index']
    new_table_generate(data, clusters, output_dir, file)
