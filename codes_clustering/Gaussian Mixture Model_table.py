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

    # 1. 최적 covariance 찾기
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

    # 2. GMM
    dpgmm = mixture.BayesianGaussianMixture(n_components=40, covariance_type=min_row_covariance).fit(mat)

    cluster_labels = dpgmm.predict(mat)

    clusters = [[] for _ in range(40)]

    for i, cluster_num in enumerate(cluster_labels):
        clusters[cluster_num].append(data.index[i])

    clusters = [sublist for sublist in clusters if sublist]

    unique_labels = []
    for _ in range(len(clusters)):
        unique_labels.append(_)

    # 2. Outlier(예정)

    # 3. Save CSV
    # columns = ['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index']
    new_table_generate(data, clusters, output_dir, file)
