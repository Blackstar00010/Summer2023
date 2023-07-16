from _table_generate import *
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# 1. 파일 불러오기
input_dir = '../files/Clustering/PCA(1-48)'
output_dir = '../files/Clustering/Gaussian_Mixture_Model'
Gaussian_Mixture_Model = sorted(filename for filename in os.listdir(input_dir))


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


# 2. CSV 파일 하나에 대해서 각각 실행
for file in Gaussian_Mixture_Model:
    data = read_and_preprocess_data(input_dir, file)

    # if data are lower than 5, using GMM is impossible.
    if len(data) < 5:
        continue

    # PCA 주성분 데이터만 가지고 있는 mat
    mat = data.values[:, 1:].astype(float)

    # 최적 mean, covariance, K찾기
    param_grid = {
        "n_components": range(1, 7),
        "covariance_type": ["spherical", "tied", "diag", "full"],
    }
    grid_search = GridSearchCV(
        GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
    )

    grid_search.fit(mat)

    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_covariance_type": "Type of covariance",
            "mean_test_score": "BIC score",
        }
    )

    cluster_labels = grid_search.predict(mat)

    unique_labels = sorted(list(set(cluster_labels)))

    clusters = [[] for _ in range(len(unique_labels))]

    for i, cluster_num in enumerate(cluster_labels):
        clusters[cluster_num].append(data.index[i])

    # 3. Save CSV
    # columns = ['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index']
    new_table_generate(data, clusters, output_dir, file)
