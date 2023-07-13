import pandas as pd
from sklearn.cluster import KMeans
from Cluster_Plot import plot_clusters

# Clusters the firms using K-Means algorithm
# Performs just one CSV file

# Read data from CSV file
data = pd.read_csv('../files/momentum/2010-01.csv', index_col = 0)
data_array = data.values  # Exclude the first column (firm names)
firm_names = data.index  # Get the first column (firm names)

# Define the number of clusters k
k_values = [5]

# Perform k-means clustering for each value of k
clusters_k = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # n_init setting to suppress warning
    kmeans.fit(data_array)  # Compute K-Means clustering
    cluster_labels = kmeans.labels_  # Label of each point (ndarray of shape)

    clusters = [[] for _ in range(k)]

    for i, cluster in enumerate(cluster_labels):
        # i: index
        # cluster: momentum value
        clusters[cluster].append(firm_names[i])

    clusters_k.append(clusters)

# Print the clusters for each k value & plot the clusters
for i, clusters in enumerate(clusters_k):
    print(f'Clusters for k = {k_values[i]}:')
    for j, firms in enumerate(clusters):
        plot_clusters(j, firms, firm_names, data_array)  # Use the imported function
    print()