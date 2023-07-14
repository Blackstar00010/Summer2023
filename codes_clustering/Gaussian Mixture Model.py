from _gmm import *
import pandas as pd
import matplotlib.pyplot as plt
from _Cluster_Plot import plot_clusters

# 1. 파일 불러오기
# data = pd.read_csv('C:/Users/김주환/Desktop/My files/PCA/2018-01.csv', header=None, index_col=[0])
data = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', header=None, index_col=[0])
firm_list = data.index[1:]
data = data[1:]
LS = data.values
mata = LS[0:, 1:]
mata = mata.astype(float)
LS = LS.astype(float)
print('Mom1+PCA')
print(LS)
print('Only PCA')
print(mata)

# 2. GMM 구현
DEBUG = True
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


# 4. GMM 결과출력
data_array = mata
firm_names = firm_list

# TODO: unique_label -> cluster_label
unique_labels = []
for i in range(1, K + 1):
    unique_labels.append(i)

# Print and plot the clusters
for i, firms in enumerate(class_indices):
    # outlier = -1 조건 맞추기 위해 TODO
    plot_clusters(unique_labels[i] - 1, firms, firm_names, data_array)  # Use the imported function
