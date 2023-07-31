from _Cluster_Plot import plot_clusters
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import *
from t_SNE import *

# 데이터 불러오기
input_dir = '../files/PCA/PCA(1-48)'
file = '2018-01.csv'
data = read_and_preprocess_data(input_dir, file)
mat = data.values[:, 1:].astype(float)

# 1. Hierachical Agglomerative
# 거리 행렬 계산
dist_matrix = pdist(mat, metric='euclidean')  # data point pair 간의 euclidean distance/firm수 combination 2
distance_matrix = squareform(dist_matrix)

# 연결 매트릭스 계산
Z = linkage(dist_matrix, method='ward')  # ward method는 cluster 간의 variance를 minimize
'''we adopt the average linkage, which is defined as the average distance between
the data points in one cluster and the data points in another cluster
논문과는 다른 부분. average method대신 ward method 사용.
'''

# 덴드로그램 시각화
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# copheric distance 계산
copheric_dis = cophenet(Z)
copheric_dis_matrix = squareform(copheric_dis)
# cophenet: dendrogram과 original data 사이 similarity을 나타내는 correlation coefficient
# 숫자가 클 수록 원본데이터와 유사도가 떨어짐. dendrogram에서 distance의미.



# Cluster k개 생성
k = 10
clusters = fcluster(Z, k, criterion='maxclust')


# 2. Outlier


def find_outliers_hac(threshold):
    cluster_distances = []
    for i in range(0, len(clusters)):
        avg_cpr_distance = sum(copheric_dis_matrix[i]) / len(clusters)
        # 각 회사별로 cophenet distance의 average distance를 구함.
        cluster_distances.append(avg_cpr_distance)

    # 클러스터링 결과 중 평균 거리 이상의 데이터 포인트를 outlier로 식별
    outliers = np.where(np.array(cluster_distances) > max(copheric_dis) * threshold)[0]
    # avg_cpr_distance가 max_cophenet distance의 alpha percentile보다 크면 outlier
    '''In our empirical study, we specify the maximum distance rather than the number of clusters K, 
    using a method similar to the method adopted for k-means clustering: 
    e is set as an α percentile of the distances between a pair of nearest data points'''
    return outliers


outliers = find_outliers_hac(0.9)

for i in range(0, len(outliers)):
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

t_SNE(mat, clusters)
