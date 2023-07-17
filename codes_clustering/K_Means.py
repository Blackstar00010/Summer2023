from _table_generate import *
from sklearn.cluster import KMeans
from _Cluster_Plot import plot_clusters

# Clusters the firms using K-Means algorithm
# Performs just one CSV file
# Read data from CSV file
input_dir = '../files/PCA/PCA(1-48)'
file = '2016-01.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)  # Exclude the first column (firm names) & Exclude MOM_1
firm_names = data.index  # Get the first column (firm names)

# Define the number of clusters k
k_values = [5, 10]


def perform_kmeans(k_values, data_array, firm_names):
    # Perform k-means clustering for each value of k
    clusters_k = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)  # n_init setting to suppress warning
        kmeans.fit(data_array)  # Compute K-Means clustering
        cluster_labels = kmeans.labels_  # Label of each point (ndarray of shape)

        clusters = [[] for _ in range(k)]

        for i, cluster in enumerate(cluster_labels):
            # i: firm index
            # cluster: cluster index
            clusters[cluster].append(firm_names[i])

        clusters_k.append(clusters)
    return clusters_k


if __name__ == "__main__":
    clusters_k = perform_kmeans(k_values, data_array, firm_names)
    # Print the clusters for each k value & plot the clusters
    for i, clusters in enumerate(clusters_k):
        print(f'Clusters for k = {k_values[i]}:')
        for j, firms in enumerate(clusters):
            plot_clusters(j, firms, firm_names, data_array)  # Use the imported function
