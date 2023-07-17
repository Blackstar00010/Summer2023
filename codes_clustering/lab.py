from _table_generate import *
from sklearn.cluster import KMeans

# Clusters the firms using K-Means algorithm
# Performs just one CSV file
# Read data from CSV file
input_dir = '../files/Clustering/PCA(1-48)'
file = '2016-01.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)  # Exclude the first column (firm names) & Exclude MOM_1
firm_names = data.index  # Get the first column (firm names)


kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
X = kmeans.fit(data_array)
cluster_labels = kmeans.labels_
print(cluster_labels)

distance=kmeans.fit_transform(data_array)
print(distance)
print(len(distance))
print(len(distance[0]))
cluster_distance_min = np.min(distance, axis=1)
print(cluster_distance_min)
cluster_distance_max = np.max(distance, axis=1)
print(cluster_distance_max)