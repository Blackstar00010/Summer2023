import os
import math
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import *
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors


class Clustering:
    def __init__(self, data):
        self.PCA_Data = data
        self.K_Mean = []
        self.DBSCAN = []
        self.Agglomerative = []
        self.Gaussian = []
        self.OPTIC = []
        self.HDBSCAN = []

    def outliers(self, K):
        data_array = self.PCA_Data.values[:, 1:].astype(float)  # Exclude the first column (firm names) & Exclude MOM_1
        firm_names = self.PCA_Data.index

        kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
        kmeans.fit(data_array)
        cluster_labels = kmeans.labels_  # Label of each point(ndarray of shape)
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

        return clust, cluster_labels

    def perform_kmeans(self, k_values) -> pd.DataFrame:
        clusters_k = []
        for k in k_values:
            n_sample = self.PCA_Data.shape[0]  # number of values in the file
            # Skip if the number of values are less than k
            if n_sample <= k_values[0]:
                continue
            clust, cluster_labels = self.outliers(k)
            clusters_k.append(clust)

        self.K_Mean = clusters_k
        return self.K_Mean

    def gmm_bic_score(self, estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    def GMM(self, threshold):

        mat = self.PCA_Data.values[:, 1:].astype(float)

        if len(mat) < 10:
            breakpoint()

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

        # Optimal Cluster
        n_components = 40
        dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=min_row_covariance).fit(mat)
        cluster_labels = dpgmm.predict(mat)

        clusters = [[] for _ in range(40)]

        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(i)

        empty_cluster_indices = [idx for idx, cluster in enumerate(clusters) if not cluster]

        n_components = n_components - len(empty_cluster_indices)

        # Outlier
        dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=min_row_covariance).fit(mat)
        cluster_labels = dpgmm.predict(mat)

        clusters = [[] for _ in range(n_components)]

        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(self.PCA_Data.index[i])

        probabilities = dpgmm.predict_proba(mat)

        cluster_prob_mean = np.mean(probabilities, axis=0)

        threshold = threshold
        outliers = []

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

        # copheric distance 계산
        copheric_dis = cophenet(Z)
        copheric_dis_matrix = squareform(copheric_dis)
        # cophenet: dendrogram과 original data 사이 similarity을 나타내는 correlation coefficient
        # 숫자가 클 수록 원본데이터와 유사도가 떨어짐. dendrogram에서 distance의미.

        # Cluster k개 생성
        k = 10
        clusters = fcluster(Z, k, criterion='maxclust')

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

    def OPTICS(self):
        data_array = self.PCA_Data.values[:, 1:].astype(float)

        labels = OPTICS(cluster_method='xi', metric='braycurtis').fit(data_array).labels_

        # Get the unique cluster labels
        unique_labels = sorted(list(set(labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(labels):
            clust[unique_labels.index(cluster_label)].append(self.PCA_Data.index[i])

        self.OPTIC = clust
        return self.OPTIC

    def perform_DBSCAN(self):
        ms = int(math.log(len(self.PCA_Data)))

        # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
        nbrs = NearestNeighbors(n_neighbors=ms + 1).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        # Sort the average distances in ascending order
        sorted_distances = np.sort(avg_distances)

        # Calculate the index for the alpha percentile (alpha)
        alpha_percentile_index = int(len(sorted_distances) * 0.9)

        eps = sorted_distances[alpha_percentile_index]

        print(ms)
        print(eps)

        cluster_labels = DBSCAN(min_samples=ms, eps=eps, metric='manhattan').fit(self.PCA_Data).labels_

        # Get the unique cluster labels
        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.PCA_Data.index[i])

        self.DBSCAN=clust
        return self.DBSCAN


def read_and_preprocess_data(input_dir, file) -> pd.DataFrame:
    data = pd.read_csv(os.path.join(input_dir, file), index_col=0)

    # Replace infinities with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    data.dropna(inplace=True)

    return data


def get_pca_data(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.transform(data), pca


def get_pd_from_pca(pca_data, cols=None):
    if cols is None:
        cols = ['pca_component_{}'.format(i + 1) for i in range(pca_data.shape[1])]
    return pd.DataFrame(pca_data, columns=cols)


def variance_ratio(pca):
    sum = np.sum(pca.explained_variance_ratio_)
    return sum


def print_variance_ratio(pca):
    print('variance_ratio: ', pca.explained_variance_ratio_)
    print('sum of variance_ratio: ', np.sum(pca.explained_variance_ratio_))


def generate_PCA_Data(data: pd.DataFrame):
    mom1 = data.values.astype(float)[:, 0]
    data_normalized = (data - data.mean()) / data.std()

    mat = data_normalized.values.astype(float)

    # mom1을 제외한 mat/PCA(2-49)
    # mat = np.delete(mat, 0, axis=1)

    # # mom49를 제외한 mat/PCA(1-48)
    mat = np.delete(mat, 48, axis=1)

    # 1. Searching optimal n_components
    if len(data) < 20:
        n_components = len(data)

    else:
        n_components = 20

    pca = PCA(n_components)
    pca.fit(mat)
    t = variance_ratio(pca)

    while t > 0.99:
        n_components -= 1
        pca = PCA(n_components)
        pca.fit(mat)
        t = variance_ratio(pca)

    while t < 0.99:
        n_components += 1
        pca = PCA(n_components)
        pca.fit(mat)
        t = variance_ratio(pca)

    # 2. PCA
    pca_mat, pca = get_pca_data(mat, n_components=n_components)
    mat_pd_pca = get_pd_from_pca(pca_mat)
    mat_pd_pca_matrix = mat_pd_pca.values

    # Original Mom1 Combining
    first_column = mom1
    first_column_matrix = np.array(first_column).reshape(-1, 1)
    combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
    df_combined = pd.DataFrame(combined_matrix)
    df_combined.index = data.index

    return df_combined


class LS_Table:

    def __init__(self, data: pd.DataFrame):
        self.PCA_Data = data

    def new_table_generate(self, Cluster, output_dir, file):
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

        # # Save the output to a CSV file in the output directory
        # LS_table.to_csv(os.path.join(output_dir, file), index=False)
        print(output_dir)
        print(file)


    def reversal_table_generate(self, data, output_dir, file):
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

        # # Save the output to a CSV file in the output directory
        # LS_table_reversal.to_csv(os.path.join(output_dir, file), index=False)

        print(output_dir)
        print(file)


class Companies:
    def __init__(self, code_list: list):
        """
        :param code_list: list of RIC codes
        """
        self.ric_codes = code_list
        self.isins = []
        self.cusips = []
        self.sedols = []
        self._data_list = []
        self._raw_data_list = []

    def fetch_symb(self, symb_type: str) -> pd.DataFrame:
        """
        Returns translation dataframe of ISIN/CUSIP/SEDOL.
        :param symb_type: 'ISIN', 'CUSIP' or 'SEDOL'
        :return: pd.DataFrame of columns 'RIC' and 'ISIN'/'CUSIP'/'SEDOL'
        """
        result = ek.get_symbology(self.ric_codes, from_symbol_type='RIC', to_symbol_type=symb_type)
        result = result.reset_index()
        result = result.rename(columns={'index': 'RIC'})
        try:
            symb = result[['RIC', symb_type]]
        except KeyError:
            result[symb_type] = ''
            symb = result[['RIC', symb_type]]
        except ek.EikonError as eke:
            print('Error code: ', eke.code)
            symb = self.fetch_symb(symb_type=symb_type)
        return symb

    def fetch_isin(self) -> pd.DataFrame:
        """
        Returns translation dataframe of ISIN Codes. Recommend using `fetch_symb('ISIN')` instead.
        :return: pd.DataFrame of columns `RIC` and `ISIN`
        """
        self.isins = self.fetch_symb('ISIN')
        return self.isins

    def fetch_cusip(self) -> pd.DataFrame:
        """
        Returns translation dataframe of CUSIP Codes. Recommend using `fetch_symb('CUSIP')` instead.
        :return: pd.DataFrame of columns `RIC` and `CUSIP`
        """
        self.cusips = self.fetch_symb('CUSIP')
        return self.cusips

    def fetch_sedol(self) -> pd.DataFrame:
        """
        Returns translation dataframe of SEDOL Codes. Recommend using `fetch_symb('SEDOL')` instead.
        :return: pd.DataFrame of columns `RIC` and `SEDOL`
        """
        self.sedols = self.fetch_symb('SEDOL')
        return self.sedols

    def fetch_data(self, tr_list, start='1983-01-01', end='2023-06-30', period='FY') -> pd.DataFrame:
        """
        Fetches and returns data in pandas DataFrame without error.
        This DataFrame is stored in this instance, so to view previous fetches, use show_history() function.
        :param tr_list: list-like data of TR fields (e.g. ['TR.SharesOutstanding', 'TR.Revenue']
        :param start: the first date to fetch data, in the format 'YYYY-MM-DD' (e.g. '1983-01-01')
        :param end: the last date to fetch data, in the format 'YYYY-MM-DD' (e.g. '2020-12-31')
        :param period: period of which the data is fetched. 'FY' by default (e.g. 'FY', 'FS', 'FQ', 'daily')
        :return: DataFrame that contains RIC codes in 'Instrument' column and other data in columns named after TR field names
        """
        if len(start.split('-')) != 3 or len(end.split('-')) != 3:
            raise ValueError('start and end values should be given in the format of "YYYY-MM-DD". ')
        if period not in ['FY', 'FS', 'FQ', 'daily']:
            raise ValueError('period value should be given as either "FY", "FS", "FQ", or "daily". ')
        if type(tr_list) != list:
            tr_list = [item for item in tr_list]

        tr_and_date_list = tr_list + [item + '.CALCDATE' for item in tr_list]
        tr_and_date_list.sort()
        fields = []
        [fields.append(ek.TR_Field(tr_item)) for tr_item in tr_and_date_list]

        datedict = {"SDate": start, "EDate": end, 'Curn': 'GBP', 'Period': 'period' + '0', 'Frq': period}

        try:
            df, err = ek.get_data(self.ric_codes, fields, parameters=datedict, field_name=True)
            self._raw_data_list.append(df)
            for col in df.columns:
                if col.count('.') < 2:  # if not calcdate
                    continue
                df[col] = df[col].astype(str)
                df.loc[:, col] = df.loc[:, col].str[:10]
                df[col].replace("<NA>", float('NaN'), inplace=True)
                df[col].replace("", float('NaN'), inplace=True)
            self._data_list.append(df)
            return df
        except ek.EikonError as eke:
            print('Error code: ', eke.code)
            if eke.code in _not_my_fault:
                # sleep(1)
                return self.fetch_data(tr_list, start=start, end=end, period=period)
            elif eke.code == 429:
                beep()
                raise RuntimeError('Code 429: reached API calls limit')
            else:
                raise RuntimeError('An error occurred; read the message above!')

    def fetch_price_data(self, start='2010-01-01', end='2023-06-30') -> pd.DataFrame:
        """
        Fetches and returns ohlc+v data in pandas DataFrame without error.
        This DataFrame is stored in this instance, so to view previous fetches, use show_history() function.
        :param start: the first date to fetch data, in the format 'YYYY-MM-DD' (e.g. '1983-01-01')
        :param end: the last date to fetch data, in the format 'YYYY-MM-DD' (e.g. '2020-12-31')
        :return: DataFrame that contains RIC codes in 'Instrument' column and other data in columns named after TR field names
        """
        if len(start.split('-')) != 3 or len(end.split('-')) != 3:
            raise ValueError('start and end values should be given in the format of "YYYY-MM-DD". ')

        tr_list = ['TR.OPENPRICE', 'TR.HIGHPRICE', 'TR.LOWPRICE', 'TR.CLOSEPRICE', 'TR.Volume']
        tr_and_date_list = tr_list + [item + '.CALCDATE' for item in tr_list]
        tr_and_date_list.sort()
        fields = []
        [fields.append(ek.TR_Field(tr_item)) for tr_item in tr_and_date_list]

        datedict = {"SDate": start, "EDate": end, 'Curn': 'GBP'}

        df, err = ek.get_data(self.ric_codes, fields, parameters=datedict, field_name=True)
        self._raw_data_list.append(df)
        for col in df.columns:
            if col[-9:] != '.CALCDATE':
                continue
            df[col] = df[col].astype(str)
            df.loc[:, col] = df.loc[:, col].str[:10]
            df[col].replace("<NA>", float('NaN'), inplace=True)
            df[col].replace("", float('NaN'), inplace=True)
        self._data_list.append(df)
        return df

    def get_history(self, index=None, raw=False) -> list:
        """
        Returns previous fetch(es) of data.
        :param index: The indices of history (e.g. -1 -> last fetch, 0 -> first fetch, [0, -1] -> first and last fetch, None -> all)
        :param raw: True if you want to fetch the history of raw data
        :return: list of dataframe(s)
        """
        ret_list = self._raw_data_list if raw else self._data_list
        if index is None:
            return ret_list
        elif type(index) is list:
            return [ret_list[i] for i in index]
        elif type(index) is int:
            return [ret_list[index]]

    def comp_specific_data(self, ric_code, raw=False) -> pd.DataFrame:
        """
        Returns a DataFrame whose Instrument column is ric_code from the last fetch
        :param ric_code: the code of the firm to get
        :param raw: True if you want to fetch from the last raw data
        :return: DataFrame whose Instrument column is ric_code from the last fetch
        """
        if len(self.get_history(raw=raw)) < 1:
            raise IndexError('No data has been fetched yet.')
        last_df = self.get_history(-1, raw=raw)[0]
        return last_df[last_df['Instrument'] == ric_code]

    def set_history(self, dataframes, raw=False) -> None:
        """
        Sets the list of fetch history to the given dataframes.
        :param dataframes: the list to set history as.
        :param raw: True if you want to set the history of raw data
        :return: None
        """
        if type(dataframes) is list:
            self._data_list = dataframes if not raw else self._data_list
            self._raw_data_list = dataframes if raw else self._raw_data_list
        elif type(dataframes) is pd.DataFrame:
            self._data_list = [dataframes] if not raw else self._data_list
            self._raw_data_list = [dataframes] if raw else self._raw_data_list
        else:
            raise TypeError("The parameter given to set_history() should be a list or a pd.DataFrame.")

    def clear_history(self, raw=False) -> None:
        """
        Clears all data history
        :param raw: True if you want to clear the history of raw data
        :return: None
        """
        self.set_history([], raw=raw)
