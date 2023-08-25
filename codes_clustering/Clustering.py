import math

import pandas as pd

from PCA_and_ETC import *
from sklearn.cluster import *
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import *
from scipy.spatial.distance import pdist
from scipy.spatial import distance


class Clustering:
    def __init__(self, data: pd.DataFrame):
        self.test = None
        self.PCA_Data = data
        self.index = data.index

        self.K_Mean = []
        self.DBSCAN = []
        self.Agglomerative = []
        self.Gaussian = []
        self.OPTIC = []
        self.HDBSCAN = []
        self.meanshift = []
        self.BIRCH = []

        self.K_Mean_labels = []
        self.DBSCAN_labels = []
        self.Agglomerative_labels = []
        self.Gaussian_labels = []
        self.OPTIC_labels = []
        self.Agglomerative_labels = []
        self.HDBSCAN_labels = []
        self.meanshift_labels = []
        self.BIRCH_labels = []

        self.lab = []
        self.lab_labels = []

    def outliers(self, k_value: int, alpha: float = 0.5):
        """
        :param k_value: the number of clusters
        :param alpha: the rate at which outliers are filtered
        :return: 2D list
        """
        # Exclude the first column (firm names) & Exclude MOM_1

        # kmeans = BisectingKMeans(n_clusters=k_value, init='k-means++', n_init=10, max_iter=500,
        #                          algorithm='elkan', bisecting_strategy='largest_cluster').fit(self.PCA_Data)
        kmeans = KMeans(n_clusters=k_value, n_init=10, max_iter=500).fit(self.PCA_Data)

        distance_to_own_centroid = np.array([distance.euclidean(self.PCA_Data[i], kmeans.cluster_centers_[kmeans.labels_[i]]) for i in range(len(self.PCA_Data))])

        nearest_neighbor_distances = []
        for i in range(len(self.PCA_Data)):
            distances = [distance.euclidean(self.PCA_Data[i], self.PCA_Data[j]) for j in range(len(self.PCA_Data)) if i!=j]
            nearest_neighbor_distances.append(min(distances))

        sorted_nearest_neighbor_distances = sorted(nearest_neighbor_distances)

        epsilon = sorted_nearest_neighbor_distances[int(len(sorted_nearest_neighbor_distances) * alpha)]

        outliers = np.where(distance_to_own_centroid > epsilon)[0]

        filtered_data = np.delete(self.PCA_Data, outliers, axis=0)

        clusters_indices = [[] for _ in range(3)]
        for i, label in enumerate(kmeans.labels_):
            if i in outliers:
                continue
            clusters_indices[label].append(i)

        clusters_indices.insert(0, list(outliers))

        return clusters_indices

    # def outliers(self, k_value: int, alpha: float = 0.5):
    #     """
    #     :param k_value: the number of clusters
    #     :param alpha: the rate at which outliers are filtered
    #     :return: 2D list
    #     """
    #     self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)
    #     # Exclude the first column (firm names) & Exclude MOM_1
    #
    #     kmeans = BisectingKMeans(n_clusters=k_value, init='k-means++', n_init=10, max_iter=500,
    #                              algorithm='elkan', bisecting_strategy='largest_cluster').fit(self.PCA_Data)
    #
    #     cluster_labels = kmeans.labels_  # Label of each point(ndarray of shape)
    #
    #     self.test = kmeans
    #     self.K_Mean_labels = cluster_labels
    #
    #     # Calculate outlier
    #     distance = kmeans.fit_transform(self.PCA_Data)  # Distance btw K centroid about all each points
    #     main_distance = np.min(distance, axis=1)  # Distance btw own K centroid about all each points
    #
    #     main_distance_clustering = [[] for _ in range(k_value)]
    #
    #     for cluster_num in range(k_value):
    #         distance_clustering = distance[cluster_labels == cluster_num]
    #         for i in range(len(distance_clustering)):
    #             main_distance_clustering[cluster_num].append(distance_clustering[i][cluster_num])
    #
    #     # max distance in own cluster
    #     for i, cluster in enumerate(main_distance_clustering):
    #         if not cluster:
    #             continue
    #         main_distance_clustering[i] = np.max(cluster)
    #
    #     max_main_distance_clustering = main_distance_clustering
    #
    #     clusters = [[] for _ in range(k_value)]  # Cluster별 distance 분류
    #     clusters_index = [[] for _ in range(k_value)]  # Cluster별 index 분류
    #
    #     for i, cluster_num in enumerate(cluster_labels):
    #         clusters[cluster_num].append(main_distance[i])
    #         clusters_index[cluster_num].append(self.index[i])
    #
    #     outliers = [[] for _ in range(k_value)]  # Cluster별 outliers' distance 분류
    #     for i, cluster in enumerate(clusters):
    #         if max_main_distance_clustering[i] == 0:
    #             continue
    #         for j, distance in enumerate(cluster):  # distance = 자기가 속한 클러스터 내에서 중심과의 거리, cluster별로 계산해야 함.
    #             if distance / max_main_distance_clustering[i] >= alpha:
    #                 # distance / 소속 cluster 점들 중 중심과 가장 먼 점의 거리 비율이 alpha 이상이면 outlier 분류
    #                 outliers[i].append(distance)
    #
    #     outliers_index = []  # Cluster별 outliers's index 분류
    #     for i, cluster in enumerate(clusters):
    #         if not outliers[i]:
    #             continue
    #         for k, outlier_dis in enumerate(outliers[i]):
    #             for j, distance in enumerate(cluster):
    #                 if outlier_dis == distance:
    #                     outliers_index.append(clusters_index[i][j])
    #                 else:
    #                     continue
    #
    #     outliers_index = list(set(outliers_index))
    #
    #     # a에 있는 값을 b에서 빼기
    #     for value in outliers_index:
    #         for row in clusters_index:
    #             if value in row:
    #                 row.remove(value)
    #
    #     clusters_index = [sublist for sublist in clusters_index if sublist]  # 빈 리스트 제거
    #
    #     # 1차원 리스트로 전환된 outlier를 cluster 맨앞에 저장.
    #     clusters_index.insert(0, outliers_index)
    #
    #     return clusters_index

    def perform_kmeans(self, k_value: int, alpha: float = 0.5):
        """
        :param k_value: k value to be tested
        :param alpha: the rate at which outliers are filtered
        :return: 3D list
        """
        clusters_k = []
        n_sample = self.PCA_Data.shape[0]  # number of values in the file
        # Skip if the number of values are less than k
        if n_sample <= k_value:
            k = n_sample
        clusters = self.outliers(k_value, alpha)
        clusters_k.append(clusters)

        self.K_Mean = clusters_k
        return self.K_Mean

    def perform_DBSCAN(self, threshold: float):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)
        # Exclude the first column (firm names) & Exclude MOM_1

        ms = int(math.log(len(self.PCA_Data)))

        # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
        nbrs = NearestNeighbors(n_neighbors=ms).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        # Sort the average distances in ascending order
        sorted_distances = np.sort(avg_distances)
        t = len(sorted_distances)

        # Z-Score 정규화
        mean = sorted_distances.mean()
        std = sorted_distances.std()
        z_score_normalized_array = (sorted_distances - mean) / std

        # 표준 편차 2 이상인 데이터 삭제
        filtered_sorted_distances = z_score_normalized_array[abs(z_score_normalized_array) < 2]
        k = len(filtered_sorted_distances)

        # Calculate the index for the alpha percentile (alpha)
        alpha_percentile_index = int(len(filtered_sorted_distances) * threshold)

        # filtered_sorted_distance에서 삭제한 수 만큼 sorted_distance에서도 삭제
        sorted_distances = sorted_distances[:-t + k]
        eps = sorted_distances[alpha_percentile_index]

        # for i in np.arange(0.1, 1.0, 0.1):
        #     alpha_percentile_index = int(len(filtered_sorted_distances) * i)
        #     eps = sorted_distances[alpha_percentile_index]
        #     print(eps)

        dbscan = DBSCAN(min_samples=ms, eps=eps, metric='euclidean').fit(self.PCA_Data)
        cluster_labels = dbscan.labels_

        self.test = dbscan
        self.DBSCAN_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.DBSCAN = clust
        return self.DBSCAN

    def perform_HDBSCAN(self, threshold):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)
        # Exclude the first column (firm names) & Exclude MOM_1

        dist_matrix = pdist(self.PCA_Data, metric='euclidean')
        # data point pair 간의 euclidean distance/firm수 combination 2

        # 연결 매트릭스 계산
        Z = linkage(dist_matrix, method='average')

        # cophenetic distance 계산
        coph_dists = Z[:, 2]  # Z의 두 번째 열은 cophenetic distance 값

        # 2. Outlier
        max_d = np.max(coph_dists) * threshold
        num = find_optimal_HDBSCAN_min_cluster_size(self.PCA_Data)

        # min_cluster_size는 silhouette score가 가장 높은 것 선정. 2부터 5까지 실험.
        Hdbscan = HDBSCAN(min_cluster_size=num, allow_single_cluster=True, cluster_selection_epsilon=max_d).fit(
            self.PCA_Data)
        cluster_labels = Hdbscan.labels_

        self.test = Hdbscan
        self.HDBSCAN_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.HDBSCAN = clust
        return self.HDBSCAN

    def perform_HA(self, threshold: float, draw_dendro=False):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 1. Hierachical Agglomerative
        # 거리 행렬 계산
        dist_matrix = pdist(self.PCA_Data, metric='cityblock')
        # data point pair 간의 euclidean distance/firm수 combination 2

        # 연결 매트릭스 계산
        Z = linkage(dist_matrix, method='average')
        '''we adopt the average linkage, which is defined as the average distance between
        the data points in one cluster and the data points in another cluster
        논문과는 다른 부분. average method대신 ward method 사용.
        '''

        # # 덴드로그램 시각화
        if draw_dendro:
            dendrogram(Z)
            plt.title('Dendrogram')
            plt.xlabel('Samples')
            plt.ylabel('Distance')
            plt.show()

        # cophenetic distance 계산
        coph_dists = Z[:, 2]  # Z의 두 번째 열은 cophenetic distance 값

        # 2. Outlier
        '''In our empirical study, we specify the maximum distance rather than the number of clusters K,
        using a method similar to the method adopted for k-means clustering:
        e is set as an α percentile of the distances between a pair of nearest data points'''
        # 평균 cophenetic distance의 0.4를 곱한 값을 max_d로 사용
        max_d = np.max(coph_dists) * threshold

        # cophenet: dendrogram과 original data 사이 similarity을 나타내는 correlation coefficient
        # 숫자가 클 수록 원본데이터와 유사도가 떨어짐. dendrogram에서 distance의미.

        # Cluster k개 생성
        clusters = fcluster(Z, max_d, criterion='distance')
        self.Agglomerative_labels = clusters
        unique_labels = sorted(list(set(clusters)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(clusters):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.Agglomerative = clust
        return self.Agglomerative

    def perform_GMM(self, alpha: float):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        type = find_optimal_GMM_covariance_type(self.PCA_Data)
        print(type)
        # 1. Gaussian Mixture Model
        gmm = GaussianMixture(n_components=2, init_params='k-means++', covariance_type=type).fit(self.PCA_Data)
        cluster_labels = gmm.predict(self.PCA_Data)

        self.test = gmm
        self.Gaussian_labels = cluster_labels

        clusters = [[] for _ in range(2)]

        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(i)

        clusters = [sublist for sublist in clusters if sublist]

        # Outliers
        # 각 데이터 포인트의 확률 값 계산
        probabilities = gmm.score_samples(self.PCA_Data)
        # 확률 값의 percentiles 계산 (예시로 하위 5% 이하를 outlier로 판단)
        threshold = np.percentile(probabilities, alpha)

        outliers = []
        for i, probability in enumerate(probabilities):
            if probability < threshold:
                outliers.append(i)

        # a에 있는 값을 b에서 빼기
        for value in outliers:
            for i, row in enumerate(clusters):
                if value in row:
                    clusters[i].remove(value)

        # 1차원 리스트로 전환된 outlier를 cluster 맨앞에 저장.
        clusters.insert(0, outliers)

        for i, cluster in enumerate(clusters):
            for t, num in enumerate(cluster):
                cluster[t] = self.index[num]

        self.Gaussian = clusters

        return self.Gaussian

    def perform_OPTICS(self, xi):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        ms = int(math.log(len(self.PCA_Data)))

        optics = OPTICS(xi=xi, min_samples=ms, min_cluster_size=2).fit(self.PCA_Data)
        labels = optics.labels_

        reachability = optics.reachability_[optics.ordering_]
        # Reachability 값이 threshold를 넘는 데이터 포인트 출력
        threshold = np.percentile(reachability, 95)
        outliers = np.where(reachability > threshold)[0]
        self.OPTIC_labels = labels

        for i, cluster_label in enumerate(labels):
            if i in outliers:
                labels[i] = -1

        unique_labels = sorted(list(set(labels)))
        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.OPTIC = clust

        return self.OPTIC

    def perform_meanshift(self, quantile):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(self.PCA_Data, quantile=quantile)

        ms = MeanShift(bandwidth=bandwidth, cluster_all=False).fit(self.PCA_Data)

        cluster_labels = ms.labels_
        self.test = ms
        self.meanshift_labels = cluster_labels

        # # Nearest Neighbors
        # neigh = NearestNeighbors(n_neighbors=2)
        # neigh.fit(self.PCA_Data)
        # distances, indices = neigh.kneighbors(self.PCA_Data)

        # # Outliers with low density (low number of neighbors)
        # densities = 1 / distances[:, 1]
        #
        # mean = sum(densities) / len(densities)
        # variance = sum((x - mean) ** 2 for x in densities) / len(densities)
        # std_dev = variance ** 0.5
        # normalized_lst = [(x - mean) / std_dev for x in densities]
        #
        # densities = normalized_lst
        #
        # outliers = []
        # for i, density in enumerate(densities):
        #     if density > 2 * np.std(densities):
        #         outliers.append(self.index[i])

        # Get the unique cluster labels
        unique_labels = sorted(list(set(self.meanshift_labels)))

        clusters = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(self.meanshift_labels):
            clusters[unique_labels.index(cluster_label)].append(self.index[i])

        # # a에 있는 값을 b에서 빼기
        # for value in outliers:
        #     for row in clusters:
        #         if value in row:
        #             row.remove(value)
        #
        # clusters = [sublist for sublist in clusters if sublist]  # 빈 리스트 제거
        #
        # # 빈리스트도 Outlier로 간주되기 때문에 가끔 생기는 결측값 제거.
        # outliers = [sublist for sublist in outliers if sublist]
        # # 1차원 리스트로 전환된 outlier를 cluster 맨앞에 저장.
        # clusters.insert(0, outliers)

        self.meanshift = clusters
        return self.meanshift

    def perform_BIRCH(self, percentile):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        birch = Birch(threshold=percentile, branching_factor=50).fit(self.PCA_Data)
        cluster_labels = birch.labels_

        self.test = birch

        # 클러스터의 중심
        cluster_centers = birch.subcluster_centers_

        # 클러스터 중심과의 거리 계산
        distances = np.linalg.norm(self.PCA_Data - cluster_centers[cluster_labels], axis=1)

        # 아웃라이어 여부 확인
        sorted_distances = np.sort(distances)

        # Calculate the index for the alpha percentile (alpha)
        alpha_percentile_index = int(len(sorted_distances) * 0.99)

        # filtered_sorted_distance에서 삭제한 수 만큼 sorted_distance에서도 삭제

        eps = sorted_distances[alpha_percentile_index]

        outliers = np.where(eps < sorted_distances)[0]

        cluster_labels = list(cluster_labels)

        for i, cluster_label in enumerate(cluster_labels):
            if i in outliers:
                cluster_labels[i] = -1

        self.BIRCH_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.BIRCH = clust
        return self.BIRCH


class ResultCheck:

    def __init__(self, data: pd.DataFrame):
        self.PCA_Data = data
        self.prefix = momentum_prefix_finder(self.PCA_Data)

    def ls_table(self, cluster: list, output_dir, file, save=True, raw=True):
        """
        output columns = ['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index']
        :param cluster:
        :param output_dir:
        :param file:
        :param save:
        :param raw:
        :return:
        """
        # New table with firm name, mom_1, long and short index, cluster index
        LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

        if raw:
            mom1_col_name = self.prefix + '1'
        else:
            mom1_col_name = '0'

        # consider using this
        '''
        shit = []
        for i in range(len(cluster)):
            for afirm in cluster[i]:
                shit.append([afirm, i])
        shit = pd.DataFrame(shit, columns=['Firm Name', 'Cluster Index'])
        shit = shit.set_index('Firm Name')
        shit['Momentum_1'] = self.PCA_Data[mom1_col_name]
        shit = shit.reset_index()
        shit = shit.set_index('Firm Name')
        shit = shit.sort_values(by=['Cluster Index', 'Momentum_1'], ascending=[True, False])
        spread_vec = (shit.reset_index()['Momentum_1'] -
                      shit.sort_values(by=['Cluster Index', 'Momentum_1'],
                                       ascending=[True, True]).reset_index()['Momentum_1'])
        shit = shit.reset_index()
        shit['spread'] = spread_vec
        shit['in_portfolio'] = (shit['spread'].abs() > shit['spread'].std()) * 1
        shit['LS'] = shit['in_portfolio'] * shit['spread'] / shit['spread'].abs()
        shit['LS'] = shit['LS'].fillna(0)
        shit = shit.drop(columns=['spread', 'in_portfolio'])
        '''

        all_diffs = []
        for cluster_num, firms in enumerate(cluster):
            firms_sorted = sorted(firms, key=lambda x: self.PCA_Data.loc[x, mom1_col_name])

            for i in range(len(firms_sorted) // 2):
                mom_diff = abs(self.PCA_Data.loc[firms_sorted[i], mom1_col_name] - self.PCA_Data.loc[firms_sorted[-i - 1], mom1_col_name])
                all_diffs.append(mom_diff)

        std_dev = np.std(all_diffs)

        for cluster_num, firms in enumerate(cluster):
            firms_sorted = sorted(firms, key=lambda x: self.PCA_Data.loc[x, mom1_col_name])
            long_short = [0] * len(firms_sorted)

            for i in range(len(firms_sorted) // 2):
                # Only assign long-short indices if the mom1 difference is greater than the standard deviation
                if abs(self.PCA_Data.loc[firms_sorted[i], mom1_col_name] - self.PCA_Data.loc[firms_sorted[-i - 1], mom1_col_name]) > std_dev:
                    long_short[i] = 1  # 1 to the low ones
                    long_short[-i - 1] = -1  # -1 to the high ones
                    # 0 to middle point when there are odd numbers in a cluster

            # Outlier cluster를 빼지 않는 대신 LS_Value를 0으로

            if cluster_num == 0:
                long_short = [0] * len(firms_sorted)

            # Add the data to the new table
            for i, firm in enumerate(firms_sorted):
                LS_table.loc[len(LS_table)] = [firm, self.PCA_Data.loc[firm, mom1_col_name], long_short[i], cluster_num]

        # Save the output to a CSV file in the output directory
        if save:
            LS_table.to_csv(os.path.join(output_dir, file), index=False)
            print(f'Exported to {output_dir}!')
        return LS_table

    def reversal_table(self, data: pd.DataFrame, output_dir, file, save=True):
        """

        :param data:
        :param output_dir:
        :param file:
        :param save:
        :return:
        """
        LS_table_reversal = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short'])
        firm_lists = data.index
        prefix = momentum_prefix_finder(data)
        firm_sorted = sorted(firm_lists, key=lambda x: data.loc[x, prefix+'1'])
        long_short = [0] * len(firm_sorted)
        t = int(len(firm_lists) * 0.1)
        for i in range(t):
            long_short[i] = 1
            long_short[-i - 1] = -1

        for i, firm in enumerate(firm_sorted):
            LS_table_reversal.loc[len(LS_table_reversal)] = [firm, data.loc[firm, prefix+'1'], long_short[i]]

        if save:
            # Save the output to a CSV file in the output directory
            LS_table_reversal.to_csv(os.path.join(output_dir, file), index=False)
            print(f'Exported to {output_dir}!')
        return LS_table_reversal

    def Plot_clusters_Kmean(self, clusters: list):
        firm_names = self.PCA_Data.index
        data_array = self.PCA_Data.values[:, 1:].astype(float)

        for i, cluster in enumerate(clusters):
            for j, firms in enumerate(cluster):

                if j == 0:
                    print(f'Noise: {firms}')
                    title = 'Noise'
                else:
                    print(f'Cluster {j}: {firms}')
                    title = f'Cluster {j}'

                # Plot the line graph for firms in the cluster
                for firm in firms:
                    firm_index = list(firm_names).index(firm)
                    firm_data = data_array[firm_index]

                    plt.plot(range(1, len(firm_data) + 1), firm_data, label=firm)

                plt.xlabel('Characteristics')
                plt.ylabel('Data Value')
                plt.title(title)

                # List the firm names on the side of the graph
                if len(firms) <= 10:
                    plt.legend(loc='center right')
                else:
                    plt.legend(loc='center right', title=f'Total Firms: {len(firms)}', labels=firms[:10] + ['...'])

                plt.show()

    def Plot_clusters(self, cluster: list, plttitle=None):
        firm_names = self.PCA_Data.index
        data_array = self.PCA_Data.values[:, 1:].astype(float)

        for j, firms in enumerate(cluster):
            if j == 0:
                print(f'Noise: {firms}')
                title = 'Noise'
            else:
                print(f'Cluster {j}: {firms}')
                title = f'Cluster {j}'

            # Plot the line graph for firms in the cluster
            for firm in firms:
                firm_index = list(firm_names).index(firm)
                firm_data = data_array[firm_index]

                plt.plot(range(1, len(firm_data) + 1), firm_data, label=firm)

            plt.xlabel('Characteristics')
            plt.ylabel('Data Value')
            plt.title(title) if plttitle is None else plt.title(plttitle)

            # List the firm names on the side of the graph
            if len(firms) <= 10:
                plt.legend(loc='center right')
            else:
                plt.legend(loc='center right', title=f'Total Firms: {len(firms)}', labels=firms[:10] + ['...'])

            plt.show()

    def count_outlier(self, cluster: list):
        """

        :param cluster:
        :return:
        """
        if not cluster:
            return 0

        elif not cluster[0]:
            return 0
        else:
            return len(cluster[0])
