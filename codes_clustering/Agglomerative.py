import matplotlib.pyplot as plt
from _table_generate import *
from _Cluster_Plot import plot_clusters
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

# 데이터 불러오기
input_dir = '../files/PCA/PCA(1-48)'
file = '2018-01.csv'
data = read_and_preprocess_data(input_dir, file)
mat = data.values[:, 1:].astype(float)

# 1. Hierachical Agglomerative
# Create an instance of Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=80, linkage='ward', metric='euclidean')

# Fit the model and predict the clusters
clusters = agg_clustering.fit_predict(mat)

# Calculate the linkage matrix for outlier detection
Z = linkage(mat, method='ward')


# 2. Outlier
def find_outliers_hac(Z, threshold):
    # Z is the linkage matrix
    # Each row of Z has the format [idx1, idx2, dist, sample_count]
    # So, the distances are Z[:, 2]
    linkage_distances = Z[:, 2]

    # Identify clusters that were merged at a distance greater than the threshold
    outliers = np.where(linkage_distances > threshold)[0]
    return outliers


outliers = find_outliers_hac(Z, 6)

for i in range(1, len(outliers)):
    for j in range(0, len(clusters)):
        if outliers[i] == j + 1:
            clusters[j + 1] = 0

unique_labels = sorted(list(set(clusters)))

clust = [[] for _ in unique_labels]
for i, cluster_label in enumerate(clusters):
    clust[unique_labels.index(cluster_label)].append(data.index[i])

# 3. Print and plot the clusters
for i, firms in enumerate(clust):
    plot_clusters(unique_labels[i] - 1, firms, data.index, mat)  # Use the imported function
