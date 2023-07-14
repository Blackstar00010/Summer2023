import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _Cluster_Plot import plot_clusters
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 1. 데이터 불러오기
data = pd.read_csv('../files/PCA/2018-01.csv', header=None)
firms_list = data[data.columns[0]].tolist()[1:]
data = data.set_index(data.columns[0])
data = data[1:]
LS = data.values
mat = LS[0:, 1:]

# 2. Hierachical Agglomerative 구현
# 거리 행렬 계산
dist_matrix = pdist(mat, metric='euclidean')
distance_matrix = squareform(dist_matrix)

# 연결 매트릭스 계산
Z = linkage(dist_matrix, method='ward')

# 덴드로그램 시각화
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

# 클러스터 할당
k = 100  # 예시로 클러스터 개수를 TODO: 3으로 설정
clusters = fcluster(Z, k, criterion='maxclust')


# 3. Outlier 선별
def find_outliers_hac(data, threshold):  # TODO: data 지우기
    '''

    :param data:
    :param threshold:
    :return:
    '''
    cluster_distances = []
    for i in range(0, len(clusters)):
        average_distance = sum(distance_matrix[i]) / len(distance_matrix[i])
        cluster_distances.append(average_distance)

    # 클러스터링 결과 중 평균 거리 이상의 데이터 포인트를 outlier로 식별
    outliers = np.where(np.array(cluster_distances) > threshold)[0]
    return outliers


outliers = find_outliers_hac(dist_matrix, 5.3341)
for i in range(1, len(outliers)):
    for j in range(0, len(clusters)):
        if outliers[i] == j + 1:
            clusters[j + 1] = 0

# 4. Hierachical Agglomerative 결과 출력
data_array = mat
firm_names = firms_list
unique_labels = sorted(list(set(clusters)))

clust = [[] for _ in unique_labels]
for i, cluster_label in enumerate(clusters):
    clust[unique_labels.index(cluster_label)].append(firm_names[i])

# Print and plot the clusters
for i, firms in enumerate(clust):
    plot_clusters(unique_labels[i] - 1, firms, firm_names, data_array)  # Use the imported function
