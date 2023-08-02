import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.cluster.hierarchy import *
from scipy.spatial.distance import pdist, squareform


class Clustering:
    def __init__(self, data):
        self.PCA_Data = data
        self.K_Mean = []
        self.DBSCAN = []
        self.Agglomerative = []
        self.Gaussian = []
        self.OPTIC = []
        self.HDBSCAN = []

        self.K_Mean_labels = []
        self.DBSCAN_labels = []
        self.Agglomerative_labels = []
        self.Gaussian_labels = []
        self.OPTIC_labels = []
        self.HDBSCAN_labels = []

    def outliers(self, K: int):
        '''
        :param K: int
        :return: 2D list
        '''
        data_array = self.PCA_Data.values[:, 1:].astype(float)  # Exclude the first column (firm names) & Exclude MOM_1
        firm_names = self.PCA_Data.index

        kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
        kmeans.fit(data_array)
        cluster_labels = kmeans.labels_  # Label of each point(ndarray of shape)
        self.K_Mean_labels = cluster_labels
        distance = kmeans.fit_transform(data_array)  # Distance btw K central points and each point
        cluster_distance_min = np.min(distance, axis=1)  # Distance between the point and the central point

        clusters = [[] for _ in range(K)]  # Cluster별 distance 분류

        clusters_index = [[] for _ in range(K)]  # Cluster별 index 분류
        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(cluster_distance_min[i])
            clusters_index[cluster_num].append(firm_names[i])

        outliers = [[] for _ in range(K)]  # Cluster별 outliers' distance 분류
        for i, cluster in enumerate(clusters):
            for j, distance in enumerate(cluster):  # distance = 자기가 속한 클러스터 내에서 중심과의 거리, cluster별로 계산해야 함.
                if distance == 0 or distance / max(
                        cluster) >= 0.5:  # distance / 소속 cluster 점들 중 중심과 가장 먼 점의 거리 비율이 85%이상이면 outlier 분류
                    outliers[i].append(distance)

        outliers_index = []  # Cluster별 outliers's index 분류
        for i, cluster_dis in enumerate(clusters):
            for j, outlier_dis in enumerate(outliers[i]):
                for k, firm in enumerate(cluster_dis):
                    if outlier_dis == firm:
                        outliers_index.append(clusters_index[i][k])
                    else:
                        continue

        # a에 있는 값을 b에서 빼기
        for value in outliers_index:
            for row in clusters_index:
                if value in row:
                    row.remove(value)

        clusters_index = [sublist for sublist in clusters_index if sublist]  # 빈 리스트 제거

        clust = []
        clust.append(outliers_index)
        for i in range(len(clusters_index)):
            clust.append(clusters_index[i])

        return clust

    def perform_kmeans(self, k_values: list):
        '''
        :param k_values: list
        :return: 3D list
        '''
        clusters_k = []
        for k in k_values:
            n_sample = self.PCA_Data.shape[0]  # number of values in the file
            # Skip if the number of values are less than k
            if n_sample <= k_values[0]:
                continue
            clust = self.outliers(k)
            clusters_k.append(clust)

        self.K_Mean = clusters_k
        return self.K_Mean

    def HG(self, threshold: float):
        mat = self.PCA_Data.values[:, 1:].astype(float)

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

        print(max(copheric_dis))

        # Cluster k개 생성
        k = 10
        clusters = fcluster(Z, k, criterion='maxclust')
        self.Agglomerative_labels = clusters

        # 2. Outlier

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

        for i in range(0, len(outliers)):
            for j in range(0, len(clusters)):
                if outliers[i] == j + 1:
                    clusters[j + 1] = 0

        unique_labels = sorted(list(set(clusters)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(clusters):
            clust[unique_labels.index(cluster_label)].append(self.PCA_Data.index[i])

        self.Agglomerative = clust
        return self.Agglomerative

    def perform_DBSCAN(self):
        ms = int(math.log(len(self.PCA_Data)))

        # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
        nbrs = NearestNeighbors(n_neighbors=ms + 1).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        # Sort the average distances in ascending order
        sorted_distances = np.sort(avg_distances)

        # Calculate the index for the alpha percentile (alpha)
        alpha_percentile_index = int(len(sorted_distances) * 0.92)

        eps = sorted_distances[alpha_percentile_index]

        cluster_labels = DBSCAN(min_samples=ms, eps=eps, metric='manhattan').fit(self.PCA_Data).labels_
        self.DBSCAN_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.PCA_Data.index[i])

        self.DBSCAN = clust
        return self.DBSCAN

    def gmm_bic_score(self, estimator, X):
        '''Callable to pass to GridSearchCV that will use the BIC score.
        Make it negative since GridSearchCV expects a score to maximize.
        BIC = T*ln(sum of squared residuals) + n*ln(T)
        T = number of sample / n = number of parameter
        The more residuals increase, the bigger BIC score.
        ToDo: BIC Test를 위해 필요한데 작동원리 모르겠음.'''
        return -estimator.bic(X)

    def GMM(self, threshold):

        mat = self.PCA_Data.values[:, 1:].astype(float)

        # 1. Gaussian Mixture Model
        # Optimal covariance
        param_grid = {
            "covariance_type": ["spherical", "tied", "diag", "full"],
        }
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=self.gmm_bic_score
        )
        grid_search.fit(mat)

        df = pd.DataFrame(grid_search.cv_results_)[
            ["param_covariance_type", "mean_test_score"]
        ]
        df["mean_test_score"] = -df["mean_test_score"]
        df = df.rename(
            columns={
                "param_covariance_type": "Type of covariance",
                "mean_test_score": "BIC score",
            }
        )

        min_row_index = df.iloc[:, 1].idxmin()
        min_row_covariance = df.iloc[min_row_index, 0]

        n_components = 40
        clusters = [[] for _ in range(40)]
        if len(mat) < 40:
            n_components = len(mat)
            clusters = [[] for _ in range(len(mat))]

        # Optimal Cluster

        dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=min_row_covariance).fit(mat)
        cluster_labels = dpgmm.predict(mat)

        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(i)

        empty_cluster_indices = [idx for idx, cluster in enumerate(clusters) if not cluster]

        n_components = n_components - len(empty_cluster_indices)

        # Clustering
        dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=min_row_covariance).fit(mat)
        cluster_labels = dpgmm.predict(mat)
        self.Gaussian_labels = cluster_labels

        clusters = [[] for _ in range(n_components)]

        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(self.PCA_Data.index[i])

        # Outliers
        probabilities = dpgmm.predict_proba(mat)

        cluster_prob_mean = np.mean(probabilities, axis=0)

        threshold = threshold
        outliers = []

        # if the probabilities that tht firm is in that cluster are lower than threshold, that firm is outlier.
        for i, prob_mean in enumerate(cluster_prob_mean):
            if prob_mean < threshold:
                outliers.append(clusters[i])

        # 원본에서 outlier제거.
        clusters = [x for x in clusters if x not in outliers]
        # 빈리스트도 Outlier로 간주되기 때문에 가끔 생기는 결측값 제거.
        outliers = [sublist for sublist in outliers if sublist]
        # 2차원 리스트를 1차원 리스트로 전환.
        outliers = [item for sublist in outliers for item in sublist]
        # 1차원 리스트로 전환된 outlier를 cluster 맨앞에 저장.
        clusters.insert(0, outliers)

        self.Gaussian = clusters

        return self.Gaussian

    def OPTICS(self):
        data_array = self.PCA_Data.values[:, 1:].astype(float)

        labels = OPTICS(cluster_method='xi', metric='l2').fit(data_array).labels_

        self.OPTIC_labels = labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(labels):
            clust[unique_labels.index(cluster_label)].append(self.PCA_Data.index[i])

        self.OPTIC = clust
        return self.OPTIC

    '''good
    cityblock
    euclidean
    l2
    braycurtis


    bad
    cosine
    l1
    manhattan
    correlation
    dice
    hamming
    jaccard
    kulsinski
    mahalanobis
    minkowski
    rogerstanimoto
    russellrao
    seuclidean
    sokalsneath
    yule
    canberra
    chebyshev
    sqeuclidean'''


class Result_Check_and_Save:

    def __init__(self, data: pd.DataFrame):
        self.PCA_Data = data

    def LS_Table_Save(self, Cluster, output_dir, file):
        # New table with firm name, mom_1, long and short index, cluster index
        LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

        for cluster_num, firms in enumerate(Cluster):
            if cluster_num == 0:
                continue

            # Sort firms based on momentum_1
            firms_sorted = sorted(firms, key=lambda x: self.PCA_Data.loc[x, 0])
            long_short = [0] * len(firms_sorted)
            mom_diffs = []

            for i in range(len(firms_sorted) // 2):
                # Calculate the mom1 difference for each pair
                mom_diff = abs(self.PCA_Data.loc[firms_sorted[i], 0] - self.PCA_Data.loc[firms_sorted[-i - 1], 0])
                mom_diffs.append(mom_diff)

            # Calculate the cross-sectional standard deviation of all pairs' mom1 differences
            std_dev = np.std(mom_diffs)

            for i in range(len(firms_sorted) // 2):
                # Only assign long-short indices if the mom1 difference is greater than the standard deviation
                if abs(self.PCA_Data.loc[firms_sorted[i], 0] - self.PCA_Data.loc[
                    firms_sorted[-i - 1], 0]) > std_dev:
                    long_short[i] = 1  # 1 to the low ones
                    long_short[-i - 1] = -1  # -1 to the high ones
                    # 0 to middle point when there are odd numbers in a cluster

            # Add the data to the new table
            for i, firm in enumerate(firms_sorted):
                LS_table.loc[len(LS_table)] = [firm, self.PCA_Data.loc[firm, 0], long_short[i], cluster_num]

        # Save the output to a CSV file in the output directory
        LS_table.to_csv(os.path.join(output_dir, file), index=False)
        print(output_dir)
        print(file)

    def Reversal_Table_Save(self, data, output_dir, file):
        LS_table_reversal = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short'])
        firm_lists = data.index
        firm_sorted = sorted(firm_lists, key=lambda x: data.loc[x, '1'])
        long_short = [0] * len(firm_sorted)
        t = int(len(firm_lists) * 0.1)
        for i in range(t):
            long_short[i] = 1
            long_short[-i - 1] = -1

        for i, firm in enumerate(firm_sorted):
            LS_table_reversal.loc[len(LS_table_reversal)] = [firm, data.loc[firm, '1'], long_short[i]]

        # Save the output to a CSV file in the output directory
        LS_table_reversal.to_csv(os.path.join(output_dir, file), index=False)
        print(output_dir)
        print(file)

    def Plot_clusters_Kmean(self, clusters):
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

    def Plot_clusters(self, cluster):
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
            plt.title(title)

            # List the firm names on the side of the graph
            if len(firms) <= 10:
                plt.legend(loc='center right')
            else:
                plt.legend(loc='center right', title=f'Total Firms: {len(firms)}', labels=firms[:10] + ['...'])

            plt.show()
