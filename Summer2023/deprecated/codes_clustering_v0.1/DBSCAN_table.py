from dbscan1 import *
from dbscan_paper import *
# Directory containing the input files
input_dir = '../files/PCA/PCA(1-48)_adj'
output_dir = '../../files/Clustering_adj/DBSCAN'
DBSCAN = sorted(filename for filename in os.listdir(input_dir))


for file in DBSCAN:
    # Read CSV file and delete +-inf values
    data = read_and_preprocess_data(input_dir, file)
    data_array = data.values[:, 1:].astype(float)
    firm_names = data.index

    # cluster_labels, dbscan = perform_DBSCAN(data_array, eps, min_samples)
    cluster_labels = perform_DBSCAN(data_array)

    # Get the unique cluster labels
    unique_labels = sorted(list(set(cluster_labels)))

    clust = [[] for _ in unique_labels]
    for i, cluster_label in enumerate(cluster_labels):
        clust[unique_labels.index(cluster_label)].append(data.index[i])

    new_table_generate(data, clust, output_dir, file)
