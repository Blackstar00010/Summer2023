import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# WHAT DOES THIS FILE DO? PLEASE WRITE SPECIFICALLY


# Read data from CSV file
data = pd.read_csv('../files/feature_set/1990-1.csv')
data_array = data.values[:, 1:]  # Exclude the first column (firm names)

# Define the number of clusters k
k_values = [5, 10, 50, 100, 500, 1000]

# Perform k-means clustering for each value of k
clusters_k = {}

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # n_init setting to suppress warning
    kmeans.fit(data_array)  # Compute k-means clustering
    cluster_labels = kmeans.labels_  # Label of each point (ndarray of shape)

    clusters = {i: [] for i in range(k)}  # Dictionary key-value pair
    for i, cluster in enumerate(cluster_labels):
        clusters[cluster].append(f'firm {i + 1}')

    clusters_k[k] = clusters

# Print the clusters for each k value
for k, clusters in clusters_k.items():
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