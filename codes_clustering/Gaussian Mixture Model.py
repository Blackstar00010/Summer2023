from _gmm import *
from _table_generate import *
import matplotlib.pyplot as plt
from _Cluster_Plot import plot_clusters

# 파일 불러오기
input_dir = '../files/Clustering/PCA(1-48)'
file = '2018-01.csv'
data = read_and_preprocess_data(input_dir, file)
mata = data.values[:,1:].astype(float)

# 1. Gaussian Mixture Model
Y = mata
matY = np.matrix(Y, copy=True)
K = 4
mu, cov, alpha = GMM_EM(matY, K, 100)
N = Y.shape[0]
gamma = getExpectation(matY, mu, cov, alpha)
category = gamma.argmax(axis=1).flatten().tolist()[0]

# TODO : for loop
# Cluster 생성
class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
class3 = np.array([Y[i] for i in range(N) if category[i] == 2])
class4 = np.array([Y[i] for i in range(N) if category[i] == 3])

# 리스트를 생성하여 각 클래스에 해당하는 회사이름 할당
class_indices = []
for i in range(K):
    indices = [data.index[index] for index, c in enumerate(category) if c == i][0:]
    class_indices.append(indices)

# Cluster 그래프확인
# TODO : 알아서
plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
plt.plot(class3[:, 0], class3[:, 1], 'go', label="class3")
plt.plot(class4[:, 0], class4[:, 1], 'cd', label="class4")
plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.show()

# Cluster별 갯수확인
print(len(class1))
print(len(class2))
print(len(class3))
print(len(class4))

# 3. Outlier선별(예정)


cluster_labels = []
for i in range(0, K ):
    cluster_labels.append(i)

# 4. Print and plot the clusters
for i, firms in enumerate(class_indices):
    # outlier = -1 조건 맞추기 위해 TODO
    plot_clusters(cluster_labels[i] , firms, data.index, mata)  # Use the imported function
