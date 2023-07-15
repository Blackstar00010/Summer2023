from _gmm import *
from _table_generate import *
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from _Cluster_Plot import plot_clusters

# 파일 불러오기
input_dir = '../files/Clustering/PCA(1-48)'
file = '2018-01.csv'
data = read_and_preprocess_data(input_dir, file)
mata = data.values[:, 1:].astype(float)

# 1. Gaussian Mixture Model
Y = mata
matY = np.matrix(Y, copy=True)
K = 20
mu, cov, alpha = GMM_EM(matY, K, 100)
N = Y.shape[0]
gamma = getExpectation(matY, mu, cov, alpha)
category = gamma.argmax(axis=1).flatten().tolist()[0]

unique_labels = sorted(list(set(category)))

clusters = [[] for _ in range(len(unique_labels))]

for i, cluster_num in enumerate(category):
    clusters[cluster_num].append(data.index[i])

# 2. GMM Graph
cluster_figure = []
for j in range(K):
    cluster_figure.append(np.array([Y[i] for i in range(N) if category[i] == j]))

# 색상 맵 생성
cmap = cm.get_cmap('viridis')  # 원하는 colormap 선택

for i in range(K):
    color = cmap(i / K)  # 클러스터 인덱스에 따라 색상 선택
    plt.plot(cluster_figure[i][:, 0], cluster_figure[i][:, 1], 'o', color=color, label="cluster" + str(i + 1))

plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.show()

# 3. Outlier선별(예정)

# 4. Print and plot the clusters
for i, firms in enumerate(clusters):
    plot_clusters(unique_labels[i], firms, data.index, mata)
