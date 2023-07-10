import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('../files/momentum/2010-01.csv')
data_array = data.values[:, 1:]  # Exclude the first column (firm names)
firm_names = data.values[:, 0]  # Get the first column (firm names)

# Define DBSCAN parameters
eps = 70  # Maximum distance between two samples to be considered as neighbors
min_samples = 2  # Minimum number of samples in a neighborhood for a point to be considered as a core point

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(data_array)

# Get the unique cluster labels
unique_labels = set(cluster_labels)

# Create a dictionary to store firms in each cluster
clusters = {label: [] for label in unique_labels}

# Group firms by cluster label
for i, cluster_label in enumerate(cluster_labels):
    clusters[cluster_label].append(firm_names[i])

# Print the clusters
for cluster_label, firms in clusters.items():
    print(f'Cluster {cluster_label}: {firms}')

    # Plot the line graph for firms in the cluster
    for firm in firms:
        firm_index = list(firm_names).index(firm)
        firm_data = data_array[firm_index]

        plt.plot(range(1, len(firm_data) + 1), firm_data, label=firm)

    plt.xlabel('Characteristics')
    plt.ylabel('Data Value')
    plt.title(f'Cluster {cluster_label}')

    # List the firm names on the side of the graph
    if len(firms) <= 10:
        plt.legend(loc='center right')
    else:
        plt.legend(loc='center right', title=f'Total Firms: {len(firms)}', labels=firms[:10] + ['...'])

    plt.show()

    print()