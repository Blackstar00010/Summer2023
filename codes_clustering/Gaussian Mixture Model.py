from matplotlib import pyplot as plt
import seaborn as sns
from _table_generate import *
from _Cluster_Plot import plot_clusters
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

# 파일 불러오기
input_dir = '../files/Clustering/PCA(1-48)'
file = '1992-12.csv'
data = read_and_preprocess_data(input_dir, file)
mat = data.values[:, 1:].astype(float)


# 1. GMM
def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


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

# 2. Lowest BIC
sns.catplot(
    data=df,
    kind="bar",
    x="Number of components",
    y="BIC score",
    hue="Type of covariance",
)
plt.show()

# Assuming your DataFrame is named df
min_row_index = df.iloc[:, 2].idxmin()

# Select the row with the smallest value in the 4th column
min_row = df.loc[min_row_index]
print(min_row)

# 3. Outlier(예정)

# 4. Print and plot the clusters
for i, firms in enumerate(clusters):
    plot_clusters(unique_labels[i], firms, data.index, mat)
