import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import manifold
from sklearn.decomposition import PCA


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


# PCA_Result Check
if __name__ == "__main__":
    # 파일 불러오기 및 PCA함수
    input_dir = '../files/momentum_adj'
    file = '1992-06.csv'
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
