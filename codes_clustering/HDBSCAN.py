import pandas as pd
import hdbscan
from _Cluster_Plot import plot_clusters

# Read data from CSV file
data = pd.read_csv('../files/momentum/2010-01.csv', index_col=0)
data_array = data.values  # Exclude the first column (firm names)
firm_names = data.index  # Get the first column (firm names)

# Define HDBSCAN parameters
min_cluster_size = 5  # Minimum number of points required to form a cluster

# Perform HDBSCAN clustering
hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
cluster_labels = hdbscan_cluster.fit_predict(data_array)

# Get the unique cluster labels
unique_labels = sorted(list(set(cluster_labels)))

# Create a list to store firms in each cluster
clusters = [[] for _ in unique_labels]

# Group firms by cluster label
for i, cluster_label in enumerate(cluster_labels):
    clusters[unique_labels.index(cluster_label)].append(firm_names[i])

# Print and plot the clusters
for i, firms in enumerate(clusters):
    plot_clusters(unique_labels[i], firms, firm_names, data_array)  # Use the imported function
    print()
