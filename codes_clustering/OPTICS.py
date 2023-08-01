from t_SNE import *
from _Cluster_Plot import *
from sklearn.cluster import OPTICS

# 파일 불러오기
input_dir = '../files/PCA/PCA(1-48)'
file = '2018-01.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)
firm_names = data.index

labels = OPTICS(cluster_method= 'xi', metric='euclidean').fit(data_array).labels_

if __name__ == "__main__":
    # Get the unique cluster labels
    unique_labels = sorted(list(set(labels)))

    clust = [[] for _ in unique_labels]
    for i, cluster_label in enumerate(labels):
        clust[unique_labels.index(cluster_label)].append(data.index[i])

    # 3. Print and plot the clusters
    for i, firms in enumerate(clust):
        plot_clusters(unique_labels[i], firms, data.index, data_array)  # Use the imported function

    t_SNE(data_array, labels)
