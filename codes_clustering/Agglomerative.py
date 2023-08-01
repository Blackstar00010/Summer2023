import matplotlib.pyplot as plt
from _table_generate import *
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import *

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

    # Step 3: Set the threshold distance (ε)
    epsilon = threshold

    # Step 4: Count the number of neighbors for each data point
    num_neighbors = np.zeros(len(data))
    for i in range(len(data)):
        num_neighbors[i] = np.sum(linkage_distances < epsilon)


    # # Identify clusters that were merged at a distance greater than the threshold
    # outliers = np.where(linkage_distances > threshold)[0]
    return num_neighbors


outliers = find_outliers_hac(Z, 0.3)

print(outliers)

for i in range(1, len(outliers)):
    for j in range(0, len(clusters)):
        if outliers[i] == j + 1:
            clusters[j + 1] = 0

unique_labels = sorted(list(set(clusters)))

clust = [[] for _ in unique_labels]
for i, cluster_label in enumerate(clusters):
    clust[unique_labels.index(cluster_label)].append(data.index[i])

# # 3. Print and plot the clusters
# for i, firms in enumerate(clust):
#     plot_clusters(unique_labels[i] - 1, firms, data.index, mat)  # Use the imported function

first = False
if first:
    dist_matrix = pdist(mat, metric='euclidean')

    Z = linkage(dist_matrix, method='ward')

    dendrogram(Z)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

    copheric_dis=cophenet(Z)
    copheric_dis_matrix=squareform(copheric_dis)

    #
    # for i in range(0, len(clusters)):
    #     cluster_distances = []
    #     average_distance = sum(copheric_dis_matrix[i]) / len(copheric_dis_matrix[i])
    #     print(average_distance)
    #     percentile = average_distance / max(copheric_dis)
    #     cluster_distances.append(average_distance)
    #     print(percentile)

    def find_outliers_hac(threshold):
        cluster_distances = []
        for i in range(0, len(clusters)):
            average_distance = sum(copheric_dis_matrix[i]) / len(copheric_dis_matrix[i])
            percentile=average_distance/max(copheric_dis)
            cluster_distances.append(percentile)

        # 클러스터링 결과 중 평균 거리 이상의 데이터 포인트를 outlier로 식별
        outliers = np.where(np.array(cluster_distances) > threshold)[0]
        return outliers


    clusters = fcluster(Z, 2, criterion='inconsistent')
    MR = maxRstat(Z, R, 3)
    fcluster(Z, t=0.8, criterion='monocrit', monocrit=MR)

    print(clusters)

    '''
    def find_outliers_hac(threshold):
        cluster_distances = []
        for i in range(0, len(clusters)):
            average_distance = sum(distance_matrix[i]) / len(distance_matrix[i])
            cluster_distances.append(average_distance)

        # 클러스터링 결과 중 평균 거리 이상의 데이터 포인트를 outlier로 식별
        outliers = np.where(np.array(cluster_distances) > threshold)[0]
        return outliers


    outliers = find_outliers_hac(6)'''

    '''
    def find_outliers_hac(threshold):
        cluster_distances = []
        for i in range(0, len(clusters)):
            average_distance = sum(copheric_dis_matrix[i]) / len(copheric_dis_matrix[i])
            percentile = average_distance / max(copheric_dis)
            cluster_distances.append(percentile)

        # 클러스터링 결과 중 평균 거리 이상의 데이터 포인트를 outlier로 식별
        outliers = np.where(np.array(cluster_distances) > threshold)[0]
        return outliers'''
