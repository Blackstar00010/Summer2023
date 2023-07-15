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
    print(file)
    first = True
    if first == True:
        data = read_and_preprocess_data(input_dir, file)
        # PCA 주성분 데이터만 가지고 있는 mat과 원본 Mom1을 추가로 가지고 있는 LS생성.
        LS = data.values
        mat = LS[:, 1:]

        # 4. Hierarchical Agglomerative 알고리즘 구현
        # 거리 행렬 계산
        dist_matrix = pdist(mat, metric='euclidean')
        distance_matrix = squareform(dist_matrix)

        # 연결 매트릭스 계산
        Z = linkage(dist_matrix, method='ward')

        # Cluster k개 생성
        k = 80
        clusters = fcluster(Z, k, criterion='maxclust')

    second = True
    if second == True:
        # 5. Outlier선별(예정)

        outliers = find_outliers_hac(10)

        for i in range(1, len(outliers)):
            for j in range(0, len(clusters)):
                if outliers[i] == j + 1:
                    clusters[j + 1] = 0

        print(outliers)

    third = True
    if third == True:

        unique_labels = sorted(list(set(clusters)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(clusters):
            clust[unique_labels.index(cluster_label)].append(data.index[i])

        print(clusters)
        print(clust)


        #new_table_generate(LS, clust, output_dir, file)

    # # 클러스터 내부에서 Frim Number 기준정렬 후 Rank와 Long Short Value계산하여result_matrix저장.
    # result_matrix = []
    # for key, value in list_dict.items():
    #     row = [key]
    #     sorted_data = sorted(value, key=lambda x: float(x.split(': ')[1]), reverse=True)
    #     length = len(sorted_data)
    #     # h = Rank, t = Long Short Value
    #     for i, item in enumerate(sorted_data):
    #         rank = i - length // 2
    #         h = rank + (length + 1) // 2
    #         t = 0 if length % 2 == 0 else (1 if rank > 0 else -1)
    #         result = f"{item}: {t}: {h}"
    #         row.append(result)
    #     result_matrix.append(row)
    # df = pd.DataFrame(result_matrix)
    # # df의 각 원소에 대하여 맨 뒤에 소속 Cluster를 붙임.
    # for i in range(0, 100):
    #     for j in range(0, len(df.head(i + 1).iloc[i].dropna()) - 1):
    #         df.iloc[i, j + 1] = str(df.iloc[i, j + 1]) + ':' + str(df.iloc[i, 0])
    # df = df.iloc[:, 1:]
    # # 결측값을 제거하고 pure data만 result_list에 저장
    # result_list = []
    # for row in df.values:
    #     for item in row:
    #         if pd.notnull(item):
    #             result_list.append(item)
    # # list 값 정렬 후 ":"기준 분리
    # sorted_result_list = sorted(result_list, key=lambda x: int(x.split(":")[0][4:]))
    # split_result_list = [item.split(":") for item in sorted_result_list]
    #
    # # 7. Result CSV 생성
    # # Firm Number 대신에 실제 Firm이름으로 대체.
    # output_file = os.path.join(output_dir, file)
    # df_sorted = pd.DataFrame(split_result_list)
    # df_sorted.columns = ["Firm", "Mom1", "LS", "Rank", "Cluster"]
    # df_sorted = df_sorted.iloc[:, 1:]
    # df_sorted.insert(0, 'Firm', firms_list)
    # # 회사별로 정렬된 dataframe을 다시 Cluster에 대하여 정렬.
    # df_sorted[df_sorted.columns[4]] = pd.to_numeric(df_sorted[df_sorted.columns[4]])
    # df_sorted = df_sorted.sort_values(by=df_sorted.columns[4])
    # df_sorted = df_sorted.sort_values(by=['Cluster', df_sorted.columns[3]])
    #
    # # 8. Print the download link and file path for the saved CSV file
    # # df_sorted의 열은 "Firm", "Mom1", "LS", "Rank", "Cluster"
    # df_sorted.to_csv(output_file, index=False)
