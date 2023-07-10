import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Read data from CSV file
data = pd.read_csv('../files/momentum_past/2004-01.csv')
firm_names = data.values[:, 0]  # The first column contains firm names
data_array = data.values[:, 1:].astype(float)  # Exclude the first column (firm names)

# Separate firms with only NaN values
nan_firms = np.where(np.isnan(data_array).all(axis=1))[0]
valid_data = data_array[~np.isnan(data_array).all(axis=1)]

# Define the number of clusters k
k_values = [50]

# Perform k-means clustering for each value of k
clusters_k = {}

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # n_init setting to suppress warning
    kmeans.fit(valid_data)  # Compute k-means clustering
    cluster_labels = kmeans.labels_  # Label of each point (ndarray of shape)

    clusters = {i: [] for i in range(k)}  # Dictionary key-value pair
    for i, cluster in enumerate(cluster_labels):
        clusters[cluster].append(firm_names[i])

    clusters_k[k] = clusters

# Add the NaN firms as an additional cluster
clusters_k[k][k] = [firm_names[i] for i in nan_firms]

# Print the clusters for each k value
for k, clusters in clusters_k.items():
    print(f'Clusters for k = {k}:')
    for cluster, firms in clusters.items():
        print(f'Cluster {cluster + 1}: {firms}')

        # Plot the line graph for firms in each cluster
        for firm in firms:
            firm_index = np.where(firm_names == firm)[0][0]  # Find the index of the firm in the original data
            firm_data = data_array[firm_index]

            if not np.isnan(firm_data).all():  # Skip NaN firms for plotting
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


'''
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Performs K-Means clustering using SciKit-Learn in different number of cluster
# Prints out the clusters for each value of k and creates a line graph for each cluster

# Read data from CSV file
data = pd.read_csv('../files/momentum_past/2004-01.csv')
data_array = data.values[:, 1:]  # Exclude the first column (firm names)

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
'''