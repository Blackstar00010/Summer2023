import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import manifold
from itertools import combinations
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import coint, kpss
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import BayesianGaussianMixture


def generate_PCA_Data(data: pd.DataFrame):
    '''
    :param data: momentum_data
    :return: Mom1+PCA_Data
    '''

    # mom1 save and data Normalization
    mom1 = data.values.astype(float)[:, 0]
    data_normalized = (data - data.mean()) / data.std()
    mat = data_normalized.values.astype(float)

    # mom1을 제외한 mat/PCA(2-49)
    # mat = np.delete(mat, 0, axis=1)

    # mom49를 제외한 mat/PCA(1-48)
    mat = np.delete(mat, 48, axis=1)

    # 1. Searching optimal n_components
    if len(data) < 20:
        n_components = len(data)

    else:
        n_components = 20

    pca = PCA(n_components)
    pca.fit(mat)
    t = np.sum(pca.explained_variance_ratio_)

    while t > 0.99:
        n_components -= 1
        pca = PCA(n_components)
        pca.fit(mat)
        t = np.sum(pca.explained_variance_ratio_)

    while t < 0.99:
        n_components += 1
        pca = PCA(n_components)
        pca.fit(mat)
        t = np.sum(pca.explained_variance_ratio_)

    # 2. PCA
    pca_mat = PCA(n_components=n_components).fit(data).transform(data)
    cols = ['pca_component_{}'.format(i + 1) for i in range(pca_mat.shape[1])]
    mat_pd_pca = pd.DataFrame(pca_mat, columns=cols)
    mat_pd_pca_matrix = mat_pd_pca.values

    # 3. combined mom1 and PCA data
    first_column_matrix = np.array(mom1).reshape(-1, 1)
    combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
    df_combined = pd.DataFrame(combined_matrix)
    df_combined.index = data.index
    df_combined = pd.DataFrame(df_combined)

    return df_combined


def read_and_preprocess_data(input_dir, file) -> pd.DataFrame:
    '''
    :param input_dir: '../files/momentum_adj'
    :param file: yyyy-mm.csv
    :return: DataFrame
    '''
    data = pd.read_csv(os.path.join(input_dir, file), index_col=0)

    # Replace infinities with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    data.dropna(inplace=True)

    return data


def t_SNE(title, data, cluster_labels):
    '''
    :param data: Mom1+PCA_Data
    :param cluster_labels: cluster_labels
    '''

    '''이웃 data와 유사성을 얼마나 중요하게 고려할지 정하는 척도.
    data set이 클수록 큰 perplexities 필요'''
    perplexities = [15, 20, 25]

    # t-SNE를 사용하여 2차원으로 차원 축소
    for i in range(3):
        perplexity = perplexities[i]

        tsne = manifold.TSNE(
            n_components=2,
            random_state=0,
            perplexity=perplexity,
            learning_rate="auto",
            n_iter=1000,
        )

        X_tsne = tsne.fit_transform(data)
        plt.figure(figsize=(8, 6))
        plt.suptitle("Perplexity=%d" % perplexity)
        sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='plasma')

        plt.title('t-SNE Visualization' + title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')

        # 클러스터 라벨을 추가하여 범례(legend) 표시
        handles, labels = sc.legend_elements()
        plt.legend(handles, labels)
        plt.show()
        print()


def analysis_clustering_result(data, compartive_label, control_label):
    '''
    :param data: ground data
    :param compartive_label: cluster label to know about
    :param control_label: cluster label to be compared
    :return: etc
    '''
    print(f"Homogeneity: {metrics.homogeneity_score(compartive_label, control_label):.3f}")
    print(f"Completeness: {metrics.completeness_score(compartive_label, control_label):.3f}")
    print(f"V-measure: {metrics.v_measure_score(compartive_label, control_label):.3f}")
    print(
        f"Adjusted Rand Index: {metrics.adjusted_rand_score(compartive_label, control_label):.3f}")
    print(
        "Adjusted Mutual Information:"
        f" {metrics.adjusted_mutual_info_score(compartive_label, control_label):.3f}"
    )
    print(
        f"Silhouette Coefficient: {metrics.silhouette_score(data, compartive_label):.3f}")


def find_optimal_GMM_hyperparameter(data):
    bgm = BayesianGaussianMixture()
    # 탐색할 covariance type과 n_components 설정
    param_grid = {
        "covariance_type": ["spherical", "tied", "diag", "full"]
    }

    # BIC score를 평가 지표로 하여 GridSearchCV 실행
    grid_search = GridSearchCV(bgm, param_grid=param_grid, scoring='neg_negative_likelihood_ratio')
    grid_search.fit(data)

    # 최적의 covariance type과 n_components 출력
    best_covariance_type = grid_search.best_params_["covariance_type"]
    return best_covariance_type


coin = True
if coin:
    def read_mom_data(data):
        # mom1 save and data Normalization
        mom1 = data.values.astype(float)[:, 0]
        data_normalized = (data - data.mean()) / data.std()
        mat = data_normalized.values.astype(float)

        # mom1을 제외한 mat/PCA(2-49)
        # mat = np.delete(mat, 0, axis=1)

        # mom49를 제외한 mat/PCA(1-48)
        mat = np.delete(mat, 48, axis=1)

        df_combined = pd.DataFrame(mat)
        df_combined.insert(0, 'Mom1', mom1)
        df_combined.index = data.index

        return df_combined.T


    def cointegrate(data, s1, s2):
        x = data[s1].values
        y = data[s2].values
        _, p_value, _ = coint(x, y)
        return p_value


    def process_pair(pair, data):
        pvalue = cointegrate(data, pair)

        spread = data[pair[0]] - data[pair[1]]
        adf_result = sm.tsa.adfuller(spread)
        kpss_result = kpss(spread)

        if adf_result[1] <= 0.05 and kpss_result[1] >= 0.05 and pvalue <= 0.01:
            mean_spread = spread.mean()
            std_spread = spread.std()
            z_score = (spread - mean_spread) / std_spread

            if float(z_score[0]) < -2:
                pair2 = [pair[1], pair[0]]
            elif float(z_score[0]) > 2:
                pair2 = pair
            else:
                return None

            return pair2
        else:
            return None


    def find_cointegrated_pairs(data: pd.DataFrame) -> list:
        data = data.iloc[1:, :]
        print('1')
        pairs = pd.DataFrame(combinations(data.columns, 2))  # 모든 회사 조합
        pairs['pvalue'] = pairs.apply(lambda x: cointegrate(data, x[0], x[1]), axis=1)
        pairs = pairs.drop(pairs[pairs['pvalue'] > 0.01].index)
        pairs = pairs.drop(columns=['pvalue'])
        print('Finished filtering pairs using pvalue!')

        # nC2 rows, 48 cols
        spread_df: pd.DataFrame = pairs.apply(lambda x: data[x[0]] - data[x[1]], axis=1)

        spread_df['adf_result'] = spread_df.index[spread_df.apply(lambda x: [sm.tsa.adfuller(x)][1], axis=1)]
        spread_df = spread_df.drop(spread_df[spread_df['adf_result'] > 0.05].index)
        spread_df = spread_df.drop(columns=['adf_result'])
        print('Finished filtering pairs using adf_result!')

        spread_df['kpss_result'] = spread_df.index[spread_df.apply(lambda x: [kpss(x)][1], axis=1)]
        spread_df = spread_df.drop(spread_df[spread_df['kpss_result'] < 0.05].index)
        spread_df = spread_df.drop(columns=['kpss_result'])
        print('Finished filtering pairs using kpss_result!')

        spread_sr = spread_df[spread_df.columns[0]]
        pairs['spread'] = (spread_sr - spread_sr.mean()) / spread_sr.std()
        pairs = pairs.dropna(subset=['spread'])
        print('Finished filtering pairs using normalised spread!')

        pairs = pairs.drop(pairs[pairs['spread'].abs() <= 2].index)
        pairs['pair1'] = pairs[0] * (pairs['spread'] > 0) + pairs[1] * (pairs['spread'] <= 0)
        pairs['pair2'] = pairs[0] * (pairs['spread'] <= 0) + pairs[1] * (pairs['spread'] > 0)

        print(len(pairs))

        invest_list = pairs[['pair1', 'pair2']].values.tolist()
        return invest_list


    def find_cointegrated_pairs_deprecated(data: pd.DataFrame):
        """
        Deprecated
        :param data:
        :return:
        """
        data = data.iloc[1:, :]
        invest_list = []

        pairs = list(combinations(data.columns, 2))  # 모든 회사 조합
        pairs_len = 1

        while len(pairs) != pairs_len:
            pairs_len = len(pairs)
            found_break = False
            for i, pair in enumerate(pairs):
                pvalue = cointegrate(data, pair[0], pair[1])

                if pvalue > 0.01:
                    continue

                else:
                    spread = data[pair[0]] - data[pair[1]]
                    adf_result = sm.tsa.adfuller(spread)
                    kpss_result = kpss(spread)

                    if adf_result[1] > 0.05 and kpss_result[1] < 0.05:
                        continue

                    else:
                        mean_spread = spread.mean()
                        std_spread = spread.std()
                        z_score = (spread - mean_spread) / std_spread
                        spread_value = float(z_score[0])

                        if abs(spread_value) <= 2:
                            continue


                        elif spread_value < -2:
                            pair = (pair[1], pair[0])
                            invest_list.append(pair)
                            pairs = [p for p in pairs if all(item not in pair for item in p)]
                            break

                        elif spread_value > 2:
                            pair = (pair[0], pair[1])
                            invest_list.append(pair)
                            pairs = [p for p in pairs if all(item not in pair for item in p)]
                            break

            print(len(pairs))
            print(len(invest_list))

            if found_break:
                break

        return invest_list


    def save_cointegrated_LS(output_dir, file, mom_data, invest_list):
        LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

        for cluster_num, firms in enumerate(invest_list):
            # Sort firms based on momentum_1
            long_short = [0] * 2
            long_short[0] = -1
            long_short[1] = 1
            # Add the data to the new table
            for i, firm in enumerate(firms):
                LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], long_short[i], cluster_num]

        firm_list_after = list(LS_table['Firm Name'])
        firm_list_before = list(mom_data.T.index)
        Missing = [item for item in firm_list_before if item not in firm_list_after]

        for i, firm in enumerate(Missing):
            LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], 0, -1]

        LS_table.sort_values(by='Cluster Index', inplace=True)

        # Save the output to a CSV file in the output directory
        LS_table.to_csv(os.path.join(output_dir, file), index=False)
        print(output_dir)

# PCA_Result Check
if __name__ == "__main__":
    # 파일 불러오기 및 PCA함수
    input_dir = '../files/momentum_adj'
    file = '1990-11.csv'
    data = read_and_preprocess_data(input_dir, file)

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
    t = np.sum(pca.explained_variance_ratio_)

    while t > 0.99:
        n_components -= 1
        pca = PCA(n_components)
        pca.fit(mat)
        t = np.sum(pca.explained_variance_ratio_)

    while t < 0.99:
        n_components += 1
        pca = PCA(n_components)
        pca.fit(mat)
        t = np.sum(pca.explained_variance_ratio_)

    # 2. PCA
    pca = PCA(n_components=n_components).fit(data)
    pca_mat = pca.transform(data)
    cols = ['pca_component_{}'.format(i + 1) for i in range(pca_mat.shape[1])]
    mat_pd_pca = pd.DataFrame(pca_mat, columns=cols)
    mat_pd_pca_matrix = mat_pd_pca.values

    # Original Mom1 Combining
    first_column_matrix = np.array(mom1).reshape(-1, 1)
    combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
    df_combined = pd.DataFrame(combined_matrix)
    df_combined.index = data.index

    # Result
    print(file)
    print("original shape:", mat.shape)
    print("transformed shape:", pca_mat.shape)
    print('variance_ratio:', pca.explained_variance_ratio_)
    print('sum of variance_ratio:', np.sum(pca.explained_variance_ratio_))
    print(mat_pd_pca)
    print(df_combined)

    # Graph after PCA
    mat_new = pca.inverse_transform(pca_mat)
    plt.scatter(mat[:, 0], mat[:, 1], alpha=0.2)
    plt.scatter(mat_new[:, 0], mat_new[:, 1], alpha=0.8)
    plt.axis('equal')
    plt.show()
