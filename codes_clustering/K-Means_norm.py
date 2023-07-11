import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('../files/momentum/2010-01.csv', index_col=0)  # Set the first column as index
data_array = data.values  # Get the data values
firm_names = data.index  # Get the firm names from the index

# Define the number of clusters k
k_values = [50]

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

# Create a DataFrame to store the new table
LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

# Populate the new table
for k, clusters in clusters_k.items():
    for cluster, firms in clusters.items():
        # Sort firms based on momentum_1
        firms_sorted = sorted(firms, key=lambda x: data.loc[x, '1'])
        momentum_1_values = [data.loc[firm, '1'] for firm in firms_sorted]

        # Normalize the momentum_1 values
        mean = np.mean(momentum_1_values)
        std = np.std(momentum_1_values)
        normalized_values = [(value - mean) / (std + 1e-10) for value in momentum_1_values]

        # Assign 1 to values greater than 0 and -1 to values less than 0
        long_short = [1 if value > 0 else -1 if value < 0 else 0 for value in normalized_values]
        # long_short = normalized_values

        # Add the data to the new table
        for i, firm in enumerate(firms_sorted):
            LS_table.loc[len(LS_table)] = [firm, data.loc[firm, '1'], long_short[i], cluster+1]

# Save the new table to a CSV file
LS_table.to_csv('../files/Clustering/K-Means/2010-01-Norm.csv', index=False)