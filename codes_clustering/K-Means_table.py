import os
from _table_generate import read_and_preprocess_data, new_table_generate
from K_Means import perform_kmeans

# Define the number of clusters k
k_values = [50]

# Directory containing the input files
input_dir = '../files/PCA/PCA(1-48)'
momentum = sorted(filename for filename in os.listdir(input_dir))

# Directory to save the output files
output_dir = '../files/Clustering/K-Means'

# Process each file
for file in momentum:
    # Read CSV file and delete +-inf values
    data = read_and_preprocess_data(input_dir, file)

    data_array = data.values[:, 1:].astype(float)  # Exclude the first column (firm names) & Exclude MOM_1
    firm_names = data.index  # Get the firm names

    n_sample = data_array.shape[0]  # number of values in the file

    # Skip if the number of values are less than k
    if n_sample <= k_values[0]:
        continue

    clusters_k = perform_kmeans(k_values, data_array, firm_names)

    # for clusters in clusters_k:
    #     new_table_generate(data, clusters, output_dir, file)
