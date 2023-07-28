import os
from _table_generate import read_and_preprocess_data, new_table_generate
from dbscan_single import perform_DBSCAN

# Directory containing the input files
input_dir = '../files/momentum_adj'
momentum = sorted(filename for filename in os.listdir(input_dir))

# Directory to save the output files
output_dir = '../files/Clustering_adj/DBSCAN'

for file in momentum:
    # Read CSV file and delete +-inf values
    data = read_and_preprocess_data(input_dir, file)

    data_array = data.values  # Exclude the first column (firm names)
    firm_names = data.index  # Get the first column (firm names)

    # Define DBSCAN parameters
    eps = 0.6805  # Maximum distance between two samples to be considered as neighbors
    min_samples = 2  # Minimum number of samples in a neighborhood for a point to be considered as a core point

    cluster_labels = perform_DBSCAN(data_array, eps, min_samples)

    # Get the unique cluster labels (excluding noise)
    unique_labels = set(label for label in cluster_labels if label != -1)

    clusters = [[] for _ in unique_labels]
    for i, label in enumerate(cluster_labels):
        if label != -1:  # Exclude noise
            clusters[label].append(firm_names[i])

    new_table_generate(data, clusters, output_dir, file)
