import math
import numpy as np
from PCA_and_ETC import *
from sklearn.cluster import *
from sklearn.neighbors import NearestNeighbors
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
        self.HDBSCAN_labels = []
        self.meanshift_labels = []
        self.BIRCH_labels = []

    def perform_kmeans(self, k_value: int, alpha: float = 0.5):
        """
        :param k_value: Number of Cluster
        :param alpha:
        :return:
        """
        n_sample = self.PCA_Data.shape[0]  # number of values in the file
        # Skip if the number of values are less than k
        if n_sample <= k_value:
            k_value = n_sample

        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # ToDo: random
        kmeans = KMeans(n_clusters=k_value, n_init=10, max_iter=500, random_state=0).fit(
            self.PCA_Data)

        distance_to_own_centroid = [distance.euclidean(self.PCA_Data[i], kmeans.cluster_centers_[kmeans.labels_[i]]) for
                                    i in range(len(self.PCA_Data))]

        cluster_labels = kmeans.labels_  # Label of each point(ndarray of shape)

        self.test = kmeans
        self.K_Mean_labels = cluster_labels

        nbrs = NearestNeighbors(n_neighbors=3, p=2).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        nearest_neighbor_distances = distances[:, 1]

        sorted_nearest_neighbor_distances = sorted(nearest_neighbor_distances)

        epsilon = sorted_nearest_neighbor_distances[int(len(sorted_nearest_neighbor_distances) * alpha)]

        # outliers = np.where(distance_to_own_centroid > epsilon)[0]

        outliers = [i for i, dist in enumerate(distance_to_own_centroid) if dist > epsilon]

        clusters_indices = [[] for _ in range(k_value)]
        for i, label in enumerate(kmeans.labels_):
            if i in outliers:
                continue
            clusters_indices[label].append(i)

        clusters_indices.insert(0, list(outliers))

        final_cluster = [[] for _ in clusters_indices]
        for i, num in enumerate(clusters_indices):
            for j in num:
                final_cluster[i].append(self.index[j])

        final_cluster = [cluster for cluster in final_cluster if cluster]
        self.K_Mean = final_cluster

    def perform_DBSCAN(self, threshold: float):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)
        # Exclude the first column (firm names) & Exclude MOM_1

        ms = int(math.log(len(self.PCA_Data)))

        # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
        # 1번째는 자기자신이니까 +1
        nbrs = NearestNeighbors(n_neighbors=ms + 1, p=1).fit(self.PCA_Data)

        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        eps = np.percentile(avg_distances, threshold * 100)

        dbscan = DBSCAN(min_samples=ms, eps=eps, metric='l1').fit(self.PCA_Data)
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

    def perform_HA(self, threshold: float):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        nbrs = NearestNeighbors(n_neighbors=3, p=1).fit(self.PCA_Data)

        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        outlier_distance = np.percentile(avg_distances, threshold * 100)

        agglo = AgglomerativeClustering(n_clusters=None, metric='l1', linkage='average',
                                        distance_threshold=outlier_distance).fit(self.PCA_Data)
        cluster_labels = agglo.labels_
        self.test = agglo

        outlier = []

        for i, avg_distance in enumerate(avg_distances):
            if avg_distance > outlier_distance:
                outlier.append(i)

        for i, cluster_label in enumerate(cluster_labels):
            if i in outlier:
                cluster_labels[i] = -1

        self.Agglomerative_labels = cluster_labels

        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.Agglomerative = clust

    def perform_HDBSCAN(self, threshold):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        nbrs = NearestNeighbors(n_neighbors=3, p=1).fit(self.PCA_Data)

        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        max_d = np.percentile(avg_distances, threshold * 100)

        # min_cluster_size는 silhouette score가 가장 높은 것 선정. 2부터 5까지 실험.
        Hdbscan = HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=max_d).fit(self.PCA_Data)
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

    def perform_OPTICS(self, threshold):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        ms = int(math.log(len(self.PCA_Data)))

        nbrs = NearestNeighbors(n_neighbors=ms + 1, p=1).fit(self.PCA_Data)

        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        eps = np.percentile(avg_distances, threshold * 100)

        optics = OPTICS(cluster_method='dbscan', min_samples=ms, eps=eps, min_cluster_size=0.1, metric='manhattan').fit(
            self.PCA_Data)
        labels = optics.labels_

        # reachability = optics.reachability_[optics.ordering_]
        # # Reachability 값이 threshold를 넘는 데이터 포인트 출력
        # threshold = np.percentile(reachability, alpha)
        # outliers = np.where(reachability > threshold)[0]
        self.test = optics
        self.OPTIC_labels = labels

        # for i, cluster_label in enumerate(labels):
        #     if i in outliers:
        #         labels[i] = -1

        unique_labels = sorted(list(set(labels)))
        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.OPTIC = clust

    def perform_BIRCH(self, threshold):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        nbrs = NearestNeighbors(n_neighbors=3, p=1).fit(self.PCA_Data)

        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        max_d = np.percentile(avg_distances, threshold * 100)

        birch = Birch(threshold=max_d, n_clusters=None).fit(self.PCA_Data)
        cluster_labels = birch.labels_

        self.test = birch

        # 클러스터의 중심
        cluster_centers = birch.subcluster_centers_

        # 클러스터 중심과의 거리 계산
        distances = np.linalg.norm(self.PCA_Data - cluster_centers[cluster_labels], axis=1)

        # 아웃라이어 여부 확인
        sorted_distances = np.sort(distances)

        epsilon = sorted_distances[int(len(sorted_distances) * 0.5)]

        outliers = np.where(sorted_distances > epsilon)[0]

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

    def perform_meanshift(self, quantile):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(self.PCA_Data, quantile=quantile)

        ms = MeanShift(bandwidth=bandwidth, cluster_all=False).fit(self.PCA_Data)

        cluster_labels = ms.labels_
        self.test = ms
        self.meanshift_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(self.meanshift_labels)))

        clusters = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(self.meanshift_labels):
            clusters[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clusters.insert(0, [])

        self.meanshift = clusters

    def perform_GMM(self, n_components: float):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        type = find_optimal_GMM_covariance_type(self.PCA_Data)

        # 1. Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, init_params='k-means++', covariance_type=type).fit(
            self.PCA_Data)
        cluster_labels = gmm.predict(self.PCA_Data)

        self.test = gmm
        self.Gaussian_labels = cluster_labels

        clusters = [[] for _ in range(n_components)]

        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(i)

        clusters = [sublist for sublist in clusters if sublist]

        # Outliers
        # 각 데이터 포인트의 확률 값 계산
        probabilities = gmm.score_samples(self.PCA_Data)
        # 확률 값의 percentiles 계산 (예시로 하위 5% 이하를 outlier로 판단)

        threshold = np.percentile(probabilities, 5)

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


class ResultCheck:

    def __init__(self, data: pd.DataFrame):
        self.PCA_Data = data
        self.prefix = momentum_prefix_finder(self.PCA_Data)
        self.table = []

    def ls_table(self, cluster: list, output_dir, file, save=True, raw=False):
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
        # '''
        clusters = []
        for i in range(len(cluster)):
            for firms in cluster[i]:
                clusters.append([firms, i])
        clusters = pd.DataFrame(clusters, columns=['Firm Name', 'Cluster Index'])
        clusters = clusters.set_index('Firm Name')
        clusters['Momentum_1'] = self.PCA_Data[mom1_col_name]
        clusters = clusters.sort_values(by=['Cluster Index', 'Momentum_1'], ascending=[True, False])
        spread_vec = (clusters.reset_index()['Momentum_1'] - clusters.sort_values(by=['Cluster Index', 'Momentum_1'], \
                                                                                  ascending=[True, True]).reset_index()[
            'Momentum_1'])
        clusters = clusters.reset_index()
        clusters['spread'] = spread_vec
        clusters['in_portfolio'] = (clusters['spread'].abs() > clusters['spread'].std()) * 1
        clusters['Long Short'] = clusters['in_portfolio'] * (-clusters['spread'] / clusters['spread'].abs())
        clusters['Long Short'] = clusters['Long Short'].fillna(0)
        clusters = clusters.drop(columns=['spread', 'in_portfolio'])
        clusters.loc[clusters['Cluster Index'] == 0, 'Long Short'] = 0
        clusters.sort_values('Cluster Index', inplace=True)
        clusters = clusters[['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index']]

        if save:
            clusters.to_csv(os.path.join(output_dir, file), index=False)
            print(f'Exported to {output_dir}!')

        # '''

        self.table = clusters

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
        firm_sorted = sorted(firm_lists, key=lambda x: data.loc[x, self.prefix + '1'])
        long_short = [0] * len(firm_sorted)
        t = int(len(firm_lists) * 0.1)
        for i in range(t):
            long_short[i] = 1
            long_short[-i - 1] = -1

        for i, firm in enumerate(firm_sorted):
            LS_table_reversal.loc[len(LS_table_reversal)] = [firm, data.loc[firm, self.prefix + '1'], long_short[i]]

        if save:
            # Save the output to a CSV file in the output directory
            LS_table_reversal.to_csv(os.path.join(output_dir, file), index=False)
            print(f'Exported to {output_dir}!')
        return LS_table_reversal

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
        all_cluster = [item for sublist in cluster for item in sublist]
        firm_len = len(all_cluster)
        if not cluster:
            return 0

        elif not cluster[0]:
            return 0
        else:
            percentile = len(cluster[0]) / firm_len
            return percentile

    def count_stock_of_traded(self):
        count_non_zero = (self.table['Long Short'] != 0).sum()
        proportion = count_non_zero / len(self.table)
        return proportion
