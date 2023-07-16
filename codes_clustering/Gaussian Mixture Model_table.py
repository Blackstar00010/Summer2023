from sklearn import mixture
from _table_generate import *
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# 파일 불러오기
input_dir = '../files/Clustering/PCA(1-48)'
output_dir = '../files/Clustering/Gaussian_Mixture_Model'
Gaussian_Mixture_Model = sorted(filename for filename in os.listdir(input_dir))


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


# CSV 파일 하나에 대해서 각각 실행
for file in Gaussian_Mixture_Model:
    data = read_and_preprocess_data(input_dir, file)

    # if data are lower than 5, using GMM is impossible.
    if len(data) < 5:
        continue

    # PCA 주성분 데이터만 가지고 있는 mat
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
    # BIC가 가장 작은 게 optimal.
    min_row_covariance = df.iloc[min_row_index, 0]

    # Optimal Cluster
    n_components = 40
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=min_row_covariance).fit(mat)
    cluster_labels = dpgmm.predict(mat)

    clusters = [[] for _ in range(40)]

    for i, cluster_num in enumerate(cluster_labels):
        clusters[cluster_num].append(i)

    empty_cluster_indices = [idx for idx, cluster in enumerate(clusters) if not cluster]

    # Cluster  40개를 전부 사용하지 않으므로 빈 리스트 갯수를 40에서 빼주면 optimal Cluster Number.
    n_components = n_components - len(empty_cluster_indices)

    # Outlier
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=min_row_covariance).fit(mat)
    cluster_labels = dpgmm.predict(mat)

    clusters = [[] for _ in range(n_components)]

    for i, cluster_num in enumerate(cluster_labels):
        clusters[cluster_num].append(data.index[i])

    # 모든 회사에 대하여 각 회사가 특정 cluster에 속할 확률을 나타냄.
    probabilities = dpgmm.predict_proba(mat)

    # cluster에 대하여 회사들이 그 cluster에서 속할 평균 확률을 계산.
    cluster_prob_mean = np.mean(probabilities, axis=0)

    threshold = 0.01
    outliers = []

    # cluster_prob_mean이 threshold보다 작다면 outlier로 간주.
    for i, prob_mean in enumerate(cluster_prob_mean):
        if prob_mean < threshold:
            outliers.append(clusters[i])

    # 원본에서 outlier제거.
    clusters = [x for x in clusters if x not in outliers]

    # 3. Save CSV
    # columns = ['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index']
    new_table_generate(data, clusters, output_dir, file)
