from sklearn import mixture
from _Cluster_Plot import plot_clusters
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from t_SNE import *

def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


def GMM(data, threshold):
    '''
    :param data: whole data
    :param threshold: outlier probability
    :return: cluster
    '''
    mat = data.values[:, 1:].astype(float)

    # 1. Gaussian Mixture Model
    # Optimal covariance
    param_grid = {
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
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
        clusters[cluster_num].append(data.index[i])

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

    return clusters


if __name__ == "__main__":
    # 파일 불러오기
    input_dir = '../files/PCA/PCA(1-48)_adj'
    file = '2022-12.csv'
    data = read_and_preprocess_data(input_dir, file)
    mat = data.values[:, 1:].astype(float)

    # 1. Gaussian Mixture Model
    # Optimal covariance
    param_grid = {
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score)
    grid_search.fit(mat)

    df = pd.DataFrame(grid_search.cv_results_)[["param_covariance_type", "mean_test_score"]]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score"})

    print(df)
    min_row_index = df.iloc[:, 1].idxmin()
    min_row_covariance = df.iloc[min_row_index, 0]
    print(min_row_covariance)

    # Optimal Cluster
    n_components = 40
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=min_row_covariance).fit(mat)
    cluster_labels = dpgmm.predict(mat)

    unique_labels = sorted(list(set(cluster_labels)))

    clusters = [[] for _ in range(40)]

    for i, cluster_num in enumerate(cluster_labels):
        clusters[cluster_num].append(i)

    empty_cluster_indices = [idx for idx, cluster in enumerate(clusters) if not cluster]

    n_components = n_components - len(empty_cluster_indices)

    # Outlier
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=min_row_covariance).fit(mat)
    cluster_labels = dpgmm.predict(mat)

    unique_labels = sorted(list(set(cluster_labels)))

    clusters = [[] for _ in range(n_components)]

    for i, cluster_num in enumerate(cluster_labels):
        clusters[cluster_num].append(data.index[i])

    probabilities = dpgmm.predict_proba(mat)

    cluster_prob_mean = np.mean(probabilities, axis=0)

    threshold = 0.1
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

    # 4. Print and plot the clusters
    for i, firms in enumerate(clusters):
        plot_clusters(unique_labels[i] - 1, firms, data.index, mat)

    t_SNE(mat, cluster_labels)

'''
바꾼 이유
GMM-EM을 사용하면 Cluster 수를 지정해줘야 하기 때문에 최적 Cluster 못찾음.
(BayesianGaussianMixture는 Cluster 수를 데이터로부터 알아서 추론.)
outlier 구하기 위해 sklearn에 함수 사용
'''
