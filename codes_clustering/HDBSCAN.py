import pandas as pd
import hdbscan
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('../files/history/1990-1.csv')
data_array = data.values[:, 1:]  # Exclude the first column (firm names)

# Define HDBSCAN parameters
min_cluster_size = 3  # Minimum number of points required to form a cluster

# Perform HDBSCAN clustering
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
cluster_labels = hdbscan_clusterer.fit_predict(data_array)

# Get the unique cluster labels
unique_labels = set(cluster_labels)

# Create a dictionary to store firms in each cluster
clusters = {label: [] for label in unique_labels}

# Group firms by cluster label
for i, cluster_label in enumerate(cluster_labels):
    clusters[cluster_label].append(f'firm {i + 1}')

# Print the clusters
for cluster_label, firms in clusters.items():
    print(f'Cluster {cluster_label}: {firms}')

    # Plot the line graph for firms in the cluster
    for firm in firms:
        firm_index = int(firm.split()[1]) - 1  # Extract firm number from firm name
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
