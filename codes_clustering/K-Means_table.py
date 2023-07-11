import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define the number of clusters k
k_values = [50]

# Directory containing the input files
input_dir = '../files/momentum'

# Directory to save the output files
output_dir = '../files/Clustering/K-Means'

# Get a list of all the files in the input directory
files = os.listdir(input_dir)

# Process each file
for file in files:
    # Read data from CSV file
    data = pd.read_csv(os.path.join(input_dir, file), index_col=0)  # Set the first column as index
    data_array = data.values  # Get the data values
    firm_names = data.index  # Get the firm names

    n_sample = data_array.shape[0]
    if n_sample <= k_values[0]:
        continue
    # Perform k-means clustering for each value of k
    clusters_k = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # n_init setting to suppress warning
        kmeans.fit(data_array)  # Compute k-means clustering
        cluster_labels = kmeans.labels_  # Label of each point (ndarray of shape)

        clusters = {i: [] for i in range(k)}  # Dictionary key-value pair
        for i, cluster in enumerate(cluster_labels):
            clusters[cluster].append(firm_names[i])

        clusters_k[k] = clusters

    # New table with firm name, mom_1, long and short index, cluster index
    LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

    for k, clusters in clusters_k.items():
        for cluster, firms in clusters.items():
            # Sort firms based on momentum_1
            firms_sorted = sorted(firms, key=lambda x: data.loc[x, '1'])
            long_short = [0] * len(firms_sorted)
            for i in range(len(firms_sorted) // 2):
                long_short[i] = -1  # -1 to the high ones
                long_short[-i-1] = 1  # 1 to the low ones
                # 0 to middle point when there are odd numbers in a cluster

            # Add the data to the new table
            for i, firm in enumerate(firms_sorted):
                LS_table.loc[len(LS_table)] = [firm, data.loc[firm, '1'], long_short[i], cluster+1]

    # Save the output to a CSV file in the output directory
    LS_table.to_csv(os.path.join(output_dir, file), index=False)
