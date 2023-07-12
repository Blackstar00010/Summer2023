import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

# Print the clusters for each k value
for i, clusters in enumerate(clusters_k):
    print(f'Clusters for k = {k_values[i]}:')
    for j, firms in enumerate(clusters):
        print(f'Cluster {j + 1}: {firms}')

        # Plot the line graph for firms in each cluster
        for firm in firms:
            firm_index = list(firm_names).index(firm)
            firm_data = data_array[firm_index]

            plt.plot(range(1, len(firm_data) + 1), firm_data, label=firm)

        plt.xlabel('Characteristics')
        plt.ylabel('Data Value')
        plt.title(f'Cluster {j + 1}, k={k_values[i]}')

        # List the firm names on the side of the graph
        if len(firms) <= 10:
            plt.legend(loc='center right')
        else:
            plt.legend(loc='center right', title=f'Total Firms: {len(firms)}', labels=firms[:10] + ['...'])

        plt.show()

    print()