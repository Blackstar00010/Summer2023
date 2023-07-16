from sklearn import mixture
from _table_generate import *
from _Cluster_Plot import plot_clusters
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# 파일 불러오기
input_dir = '../files/Clustering/PCA(1-48)'
file = '1993-01.csv'
data = read_and_preprocess_data(input_dir, file)
mat = data.values[:, 1:].astype(float)


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


'''
바꾼 이유
GMM-EM을 사용하면 Cluster 수를 지정해줘야 하기 때문에 최적 Cluster 못찾음.
BayesianGaussianMixture는 Cluster 수를 데이터로부터 알아서 추론.
'''

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
print(min_row_covariance)

# 2. GMM
dpgmm = mixture.BayesianGaussianMixture(n_components=40, covariance_type=min_row_covariance).fit(mat)

cluster_labels = dpgmm.predict(mat)

clusters = [[] for _ in range(40)]
clusters2 = [[] for _ in range(40)]

for i, cluster_num in enumerate(cluster_labels):
    clusters[cluster_num].append(data.index[i])
    clusters2[cluster_num].append(i)

clusters = [sublist for sublist in clusters if sublist]
clusters2 = [sublist for sublist in clusters2 if sublist]

unique_labels = []
for _ in range(len(clusters)):
    unique_labels.append(_)

# 3. Outlier(미완)
threshold = 0.01
outliers = []

# 각 클러스터에서 이상치 판별
for cluster in clusters2:
    cluster_size = len(clusters2)
    cluster_probabilities = dpgmm.predict_proba(mat[cluster])

    # 클러스터에 속하는 각 데이터 포인트에 대해 확률 평균 계산
    cluster_prob_mean = np.mean(cluster_probabilities, axis=0)
    print(cluster_prob_mean)

    # 클러스터의 데이터 포인트 중 확률 평균이 임계값보다 작은 것을 이상치로 판별
    for i, prob_mean in enumerate(cluster_prob_mean):
        print(i)
        print(prob_mean)
        # if prob_mean < threshold:
        #     outliers.append(cluster[i])

# 이상치(outlier) 리스트 출력
print("Outliers:", outliers)

# # 4. Print and plot the clusters
# for i, firms in enumerate(clusters):
#     plot_clusters(unique_labels[i], firms, data.index, mat)
