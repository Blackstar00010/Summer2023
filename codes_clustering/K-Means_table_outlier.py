from K_Means_outlier import *

k_values = [10]
input_dir = '../files/PCA/PCA(1-48)_adj'
output_dir = '../files/Clustering_adj/K_Means_outlier'
kmeans = sorted(filename for filename in os.listdir(input_dir))


for file in kmeans:
    data = read_and_preprocess_data(input_dir, file)
    data_array = data.values[:, 1:].astype(float) # Exclude the first column (firm names) & Exclude MOM_1
    firm_names = data.index
    n_sample = data_array.shape[0]  # number of values in the file
    # Skip if the number of values are less than k
    if n_sample <= k_values[0]:
        continue

    clusters_k, kmean_data = perform_kmeans(k_values, data_array, firm_names)
    for i, clusters in enumerate(clusters_k):
        new_table_generate(data, clusters, output_dir, file)
