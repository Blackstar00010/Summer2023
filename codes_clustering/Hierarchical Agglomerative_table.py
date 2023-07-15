from _table_generate import *
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# 1. 파일 불러오기
input_dir = '../files/Clustering/PCA(1-48)'
output_dir = '../files/Clustering/Hierarchical_Agglomerative'
Hierarchical_Agglomerative = sorted(filename for filename in os.listdir(input_dir))


def find_outliers_hac(threshold):
    cluster_distances = []
    for i in range(0, len(clusters)):
        average_distance = sum(distance_matrix[i]) / len(distance_matrix[i])
        cluster_distances.append(average_distance)
    # 클러스터링 결과 중 평균 거리 이상의 데이터 포인트를 outlier로 식별
    outliers = np.where(np.array(cluster_distances) > threshold)[0]
    return outliers


# 2. CSV 파일 하나에 대해서 각각 실행
for file in Hierarchical_Agglomerative:

    data = read_and_preprocess_data(input_dir, file)
    # PCA 주성분 데이터만 가지고 있는 mat과 원본 Mom1을 추가로 가지고 있는 LS생성.
    mat = data.values[:, 1:]

    # 4. Hierarchical Agglomerative 알고리즘 구현
    # 거리 행렬 계산
    dist_matrix = pdist(mat, metric='euclidean')
    distance_matrix = squareform(dist_matrix)

    # 연결 매트릭스 계산
    Z = linkage(dist_matrix, method='ward')

    # Cluster k개 생성
    k = 80
    clusters = fcluster(Z, k, criterion='maxclust')

    # 5. Outlier선별

    outliers = find_outliers_hac(10)

    for i in range(1, len(outliers)):
        for j in range(0, len(clusters)):
            if outliers[i] == j + 1:
                clusters[j + 1] = 0

    unique_labels = sorted(list(set(clusters)))

    clust = [[] for _ in unique_labels]
    for i, cluster_label in enumerate(clusters):
        clust[unique_labels.index(cluster_label)].append(data.index[i])

    for cluster_num, firms in enumerate(clust):
        # Sort firms based on momentum_1
        firms_sorted = sorted(firms, key=lambda x: data.loc[x, '1'])
        print(firms_sorted)

    new_table_generate(data, clust, output_dir, file)
