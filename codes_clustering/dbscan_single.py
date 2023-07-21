import pandas as pd
from sklearn.cluster import DBSCAN
from _Cluster_Plot import plot_clusters
from sklearn.neighbors import NearestNeighbors
from _table_generate import *

# Clusters the firms using DBSCAN algorithm
# Performs just one CSV file
input_dir = '../files/PCA/PCA(1-48)'
file='2018-01.csv'
# Read data from CSV file
data = read_and_preprocess_data(input_dir, file)
data_array = data.values  # Exclude the first column (firm names)
firm_names = data.index  # Get the first column (firm names)

# Define DBSCAN parameters
eps = 2.404  # Maximum distance between two samples to be considered as neighbors
min_samples = 2  # Minimum number of samples in a neighborhood for a point to be considered as a core point


def perform_DBSCAN (data_array, eps, min_samples):
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
    cluster_labels = dbscan.fit_predict(data_array)

    return cluster_labels


# NearestNeighbors 모델 생성
nn_model = NearestNeighbors(n_neighbors=10, metric='manhattan')
nn_model.fit(data_array)

# 각 데이터 포인트에 대한 최근접 이웃 인덱스와 거리 계산
distances, indices = nn_model.kneighbors(data_array)

# 각 데이터 포인트의 평균 최근접 이웃 거리 계산
average_distances = np.mean(distances[:, 1:], axis=1)

cluster_labels = perform_DBSCAN(data_array, eps, min_samples)

# 클러스터 레이블 출력 및 평균 최근접 이웃 거리 출력
print("DBSCAN Cluster Labels:", cluster_labels)
print("Average Distances to MinPts Neighbors:", average_distances)

average_distance=sum(average_distances)/len(average_distances)
print(average_distance*0.5)

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

'''
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('../files/momentum/2017-01.csv')
data_array = data.values[:, 1:]  # Exclude the first column (firm names)
firm_names = data.values[:, 0]  # Get the first column (firm names)

# Define DBSCAN parameters
eps = 0.92  # Maximum distance between two samples to be considered as neighbors
min_samples = 9  # Minimum number of samples in a neighborhood for a point to be considered as a core point

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
'''