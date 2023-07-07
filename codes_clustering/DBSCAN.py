import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Read data from CSV file
data = pd.read_csv('../files/feature_set/1990-1.csv')
data_array = data.values[:, 1:]  # Exclude the first column (firm names)

# Perform DBSCAN clustering
clusters_DBSCAN = {}

dbscan = DBSCAN(eps=0.5, min_samples=5)  # Set your own parameters
dbscan.fit(data_array)  # Compute DBSCAN clustering
cluster_labels = dbscan.labels_  # Label of each point

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # Number of clusters, excluding noise if present

clusters_DBSCAN = {i: [] for i in range(n_clusters)}  # Dictionary key-value pair
for i, cluster in enumerate(cluster_labels):
    if cluster != -1:  # Exclude noise points
        clusters_DBSCAN[cluster].append(f'firm {i + 1}')

# Print the clusters for each k value
for k, clusters in clusters_DBSCAN.items():
    print(f'Clusters for k = {k}:')
    for cluster, firms in clusters.items():
        print(f'Cluster {cluster + 1}: {firms}')

        # PLot the line graph for firms in each cluster
        for firm in firms:
            firm_index = int(firm.split()[1]) - 1  # Extract firm number from firm name
            firm_data = data_array[firm_index]

            plt.plot(range(1, len(firm_data) + 1), firm_data, label=firm)

        plt.xlabel('Characteristics')
        plt.ylabel('Data Value')
        plt.title(f'Cluster {cluster + 1}, k={k}')

        # List the firm names on the side of the graph
        if len(firms) <= 10:
            plt.legend(loc='center right')
        else:
            plt.legend(loc='center right', title=f'Total Firms: {len(firms)}', labels=firms[:10] + ['...'])

        plt.show()

    print()