import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import manifold
from sklearn.mixture import *
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# turn off warning
warnings.filterwarnings("ignore")


def momentum_prefix_finder(df: pd.DataFrame):
    """

    :param df:
    :return:
    """
    possible_prefix = ['', 'Momentum', 'MOM', 'mom', 'Momentum_', 'MOM_', 'mom_', 'Mom_']
    for i in range(1, 10):
        for aposs in possible_prefix:
            if aposs + str(i) in df.columns:
                return aposs
    return ''


def merge_LS_Table(data, LS_merged_df, file):
    # Keep only the 'Firm Name' and 'Long Short' columns
    data = data[['Firm Name', 'Long Short']]

    # Change the column name into file name (ex: 1990-01)
    file_column_name = os.path.splitext(file)[0]
    data = data.rename(columns={'Long Short': file_column_name})

    if LS_merged_df.empty:
        LS_merged_df = data
    else:
        LS_merged_df = pd.merge(LS_merged_df, data, on='Firm Name', how='outer')

    return LS_merged_df


def product_LS_Table(LS_merged_df: pd.DataFrame, MOM_merged_df: pd.DataFrame, result_df: pd.DataFrame):
    # Sort LS_Value according to Firm Name
    LS_merged_df = LS_merged_df.sort_values('Firm Name')

    # I don't know the reason why'''
    # LS_merged_df = LS_merged_df.drop_duplicates(subset=LS_merged_df.columns[0], keep='first')

    # Set Firm Name column into index
    LS_merged_df.set_index('Firm Name', inplace=True)

    # 마지막 row 버리면 한칸씩 밀어버리는 것과 동치
    LS_merged_df = LS_merged_df.drop(LS_merged_df.columns[-1], axis=1)

    LS_merged_df = LS_merged_df.fillna(0)
    LS_merged_df.columns = MOM_merged_df.columns

    # Multiply only the numeric columns
    prod = MOM_merged_df * LS_merged_df
    prod = pd.DataFrame(prod)

    # prod index set to df1.index
    prod.set_index(MOM_merged_df.index, inplace=True)
    # cumulative return은 1990-02부터 2022-12이기 때문에 prod.columns=df1.columns
    prod.columns = MOM_merged_df.columns

    # Count the non-zero LS that is the number of total firm invested(395 by 1 matrix/index=Date)
    non_zero_count = LS_merged_df.astype(bool).sum()

    # sum about all rows(395 by 1 matrix/index=Date)
    column_sums = prod.sum()

    # calculate mean and make into DataFrame
    column_means = column_sums.values / non_zero_count.values
    column_means = pd.DataFrame(column_means)
    column_means.index = column_sums.index

    # Concat the means DataFrame to the result DataFrame(395 by 1 matrix->1 by 395 matrix)
    result_df = pd.concat([result_df, column_means.T], ignore_index=True)

    return result_df


def save_and_plot_result(clustering_name, result_df: pd.DataFrame, file_names, FTSE=False, apply_log=True):
    # Add a new column to the result DataFrame with the file names
    result_df['Clustering Method'] = file_names
    # result_df.set_index('Clustering Method', inplace=True)

    # Separate the 'Clustering Method' column from the date columns
    clustering_method = result_df['Clustering Method']
    date_columns_df = result_df.drop('Clustering Method', axis=1)

    # Convert the date columns to datetime format and sort them
    date_columns_df.columns = pd.to_datetime(date_columns_df.columns, errors='coerce')
    date_columns_df = date_columns_df.sort_index(axis=1)

    # Concat the 'Clustering Method' column back with the sorted date columns
    result_df = pd.concat([clustering_method, date_columns_df], axis=1)
    result_df.set_index('Clustering Method', inplace=True)

    if FTSE:
        file_names.append('FTSE 100')

        # benchmark return merge with result_df
        file = '../files/ftse_return.csv'
        df = pd.read_csv(file)
        df = df.iloc[1:]
        df = df.iloc[0:, 2:]

        df.columns = result_df.columns[0:-7]  # columns name should be same with result_df
        result_df = pd.concat([result_df.iloc[:, 0:-7], df], axis=0)  # add monthly_return right below result_df

    result_df.index = file_names
    result_df = result_df.astype(float)  # set data type as float(df.value was str actually.)
    result_df = result_df.fillna(0)

    result_df.T.describe().to_csv(os.path.join('../files/result/', f'{clustering_name}_statistcs_original.csv'),
                                  index=True)
    result_df.to_csv(os.path.join('../files/result/', f'{clustering_name}_result_original.csv'), index=True)

    # Add 1 to all data values
    result_df.iloc[:, 0:] = result_df.iloc[:, 0:] + 1

    # transform into log scale
    result_df.iloc[:, 0:] = np.log(result_df.iloc[:, 0:]) if apply_log else result_df.iloc[:, 0:]

    # drop irrational data ( larger than criterion )
    criterion = np.log(1.5) if apply_log else 2
    result_df = result_df.applymap(lambda x: float('NaN') if abs(x) > criterion else x)
    result_df = result_df.applymap(
        lambda x: float('NaN') if x < 1 - 1 / criterion else x) if not apply_log else result_df
    result_df = result_df.fillna(method='ffill', axis=1)

    result_modified = pd.DataFrame(
        index=['count', 'annual return mean', 'annual return std', 'monthly return min', 'monthly return max'],
        columns=result_df.index)
    for i in range(len(result_modified.columns)):
        result_modified.iloc[0, i] = len(result_df.columns)
        result_modified.iloc[1, i] = np.mean(result_df.iloc[i, :]) * 12
        annual_return = np.array(result_df.iloc[i, :]) * 12
        result_modified.iloc[2, i] = np.std(annual_return)
        result_modified.iloc[3, i] = np.min(result_df.iloc[i, :])
        result_modified.iloc[4, i] = np.max(result_df.iloc[i, :])
    # result_modified.iloc[1, :] = result_modified.iloc[1, :] * len(result_df.iloc[1, :]) / 12

    sharpe_ratio = pd.DataFrame(index=['Sharpe ratio'], columns=result_modified.columns)

    for i in range(len(result_modified.columns)):
        sharpe_ratio.iloc[0, i] = result_modified.iloc[1, i] / result_modified.iloc[2, i]
    result_modified = pd.concat([result_modified, sharpe_ratio], axis=0)

    MDD = pd.DataFrame(index=['Maximum drawdown'], columns=result_modified.columns)

    for i in range(len(result_df.index)):
        for j in range(1):
            row = result_df.iloc[i, :]
            cumulative_returns = (1 + row).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            MDD.iloc[j, i] = max_drawdown

    result_modified = pd.concat([result_modified, MDD], axis=0)

    result_modified.to_csv(os.path.join('../files/result/', clustering_name + '_statistcs_modified.csv'), index=True)
    result_df.to_csv(os.path.join('../files/result/', clustering_name + '_result_modified.csv'), index=True)

    # Calculate the cumulative product
    result_df.iloc[:, :] = result_df.iloc[:, :].cumsum(axis=1) if apply_log else result_df.iloc[:, :].cumprod(
        axis=1)

    Plot = True
    if Plot:
        plt.figure(figsize=(10, 6))

        for i in range(len(result_df)):
            plt.plot(result_df.columns[1:-7], result_df.iloc[i, 1:-7].fillna(method='ffill'),
                     label=result_df.iloc[i, 0])

        plt.title('RETURN')
        plt.xlabel('Date')
        plt.ylabel('cumulative Value')
        # plt.xticks(rotation=45)
        plt.legend(result_df.index)  # Add a legend to distinguish different lines
        plt.tight_layout()
        plt.show()

        # Plot a graph for each row
        for i in range(len(result_df)):
            plt.figure(figsize=(10, 6))
            plt.plot(result_df.columns[1:-7], result_df.iloc[i, 1:-7].fillna(method='ffill'))
            plt.title(result_df.index[i])
            plt.xlabel('Date')
            plt.ylabel('Average Value')
            plt.xticks(rotation=45)
            plt.tight_layout()

            plt.show()


def save_cluster_info(clustering_name, stat_list: list, file_names):
    file_names.remove('FTSE 100')
    stat_df = pd.DataFrame(stat_list,
                           index=file_names,
                           columns=['Number of clusters', 'Number of stock in clusters',
                                    'Number of outliers', 'Number of stock traded']).T

    stat_df.to_csv(os.path.join('../files/result/', f'{clustering_name}_cluster_info.csv'), index=True)


def generate_PCA_Data(data: pd.DataFrame):
    """
    :param data: momentum_data
    :return: Mom1+PCA_Data
    """
    prefix = momentum_prefix_finder(data)

    # mom1 save and data Normalization
    mom1 = data.astype(float).loc[:, prefix + '1']
    data_normalized = (data - data.mean()) / data.std()
    mat = data_normalized.astype(float)
    # mom1을 제외한 mat/PCA(2-49)
    # mat = np.delete(mat, 0, axis=1)

    # mom49를 제외한 mat/PCA(1-48)
    mat = mat.drop(columns=[prefix + '49'])
    mat = mat.dropna(how='all', axis=1)

    # 1. Searching optimal n_components
    n_components = min(len(data), 20)

    pca = PCA(n_components)
    pca.fit(mat)
    total_variance = np.sum(pca.explained_variance_ratio_)

    while total_variance > 0.99:
        n_components -= 1
        pca = PCA(n_components)
        pca.fit(mat)
        total_variance = np.sum(pca.explained_variance_ratio_)

    while total_variance < 0.99:
        n_components += 1
        pca = PCA(n_components)
        pca.fit(mat)
        total_variance = np.sum(pca.explained_variance_ratio_)

    # 2. PCA
    pca_mat = PCA(n_components=n_components).fit(data).transform(data)
    cols = [f'pca_component_{i + 1}' for i in range(pca_mat.shape[1])]
    mat_pd_pca = pd.DataFrame(pca_mat, columns=cols)
    mat_pd_pca_matrix = mat_pd_pca.values

    # 3. combined mom1 and PCA data
    first_column_matrix = np.array(mom1).reshape(-1, 1)
    combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
    df_combined = pd.DataFrame(combined_matrix)
    df_combined.columns = df_combined.columns.astype(str)
    df_combined.index = data.index

    return df_combined


def read_and_preprocess_data(input_dir, file) -> pd.DataFrame:
    """
    Only for reading YYYY-MM.csv files. Recommend using smart_read() for more general cases.
    :param input_dir: '../files/momentum_adj'
    :param file: YYYY-MM.csv
    :return: DataFrame
    """
    return smart_read(os.path.join(input_dir, file), include_index=True)


def smart_read(filename: str, include_index=False) -> pd.DataFrame:
    """

    :param filename:
    :param include_index: True if the file already has index column
    :return:
    """
    df = pd.read_csv(filename, index_col=0) if include_index else pd.read_csv(filename)

    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    df.dropna(inplace=True)

    return df


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


def find_optimal_GMM_covariance_type(data):
    bgm = GaussianMixture()
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
