# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import seaborn as sns; sns.set_theme(color_codes=True)

# Please mount your drive.


# Importing the dataset using pandas dataframe
df = pd.read_csv('../files/feature_set/*result.csv', header=None, index_col=[0])
df

df.head(1001)

mat = df.values
mat = mat[1:, :]
mat = mat.astype(float)
print(mat)
print(mat[0:,0][130])

mat.shape

"""# **Hierarchical Agglomerative Clustering**"""

# 거리 행렬 계산
dist_matrix = pdist(mat, metric='euclidean')

# 연결 매트릭스 계산
Z = linkage(dist_matrix, method='ward')

# 덴드로그램 시각화
dendrogram(Z)

plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()


# 적절한 클러스터 개수 선택
# 덴드로그램을 분석하여 적절한 클러스터 개수를 결정합니다.

# 클러스터 할당
k = 100  # 예시로 클러스터 개수를 3으로 설정
clusters = fcluster(Z, k, criterion='maxclust')

# 클러스터 할당 결과 출력
print(dist_matrix)
print("클러스터 할당 결과:")
print(clusters)

# # Agglomerative Clustering 모델 생성 및 학습
# agg_clustering = AgglomerativeClustering(n_clusters=3)
# agg_clustering.fit(mat)

# # 클러스터 할당 결과 확인
# labels = agg_clustering.labels_
# print("클러스터 할당 결과:", labels)

list_dict = {}
for i in range(k):
     list_dict[i+1] = []

for i in range(0,1000):
    for j in range(0,k):
        if clusters[i]==j+1:
          list(list_dict.values())[j].append('firm'+str(i+1)+': '+str(mat[0:,0][i]))

for key, value in list_dict.items():
    print(value)
    print(key)

# first_value = list_dict[1]

# print(first_value)

# data_list = ['firm131: 9.496533288558314', 'firm153: 6.529806037121606', 'firm557: 8.654403023361212', 'firm584: 6.282734944482915', 'firm681: 4.601568482825176', 'firm780: 6.829162649981441', 'firm993: 7.245752710476307']

# for key, value in list_dict.items():
#     print(value)
#     print(key)
#     sorted_data = sorted(data_list, key=lambda x: float(x.split(': ')[1]), reverse=True)
#     length = len(sorted_data)
#     for i, item in enumerate(sorted_data):
#       rank = i - length // 2
#       if rank==0:
#           rank = 0
#       if rank > 0:
#           rank = 1
#       elif rank < 0:
#           rank = -1

#     print(f"{item}: {rank}")

for key, value in list_dict.items():
    print(f"Cluster: {key}")
    sorted_data = sorted(value, key=lambda x: float(x.split(': ')[1]), reverse=True)
    length = len(sorted_data)
    for i, item in enumerate(sorted_data):
        rank = i - length // 2
        if rank == 0:
            rank = 0
        if rank > 0:
            rank = 1
        elif rank < 0:
            rank = -1
        print(f"{item}: {rank}")

data_list = ['firm131: 9.496533288558314', 'firm153: 6.529806037121606', 'firm557: 8.654403023361212', 'firm584: 6.282734944482915', 'firm681: 4.601568482825176', 'firm780: 6.829162649981441', 'firm993: 7.245752710476307']

sorted_data = sorted(data_list, key=lambda x: float(x.split(': ')[1]), reverse=True)

for item in sorted_data:
    print(item)

# Extracting the useful features from the dataset
plt.scatter(mat[:,0],mat[:,1])

plt.show()

linked = linkage(mat, 'single')
dendrogram(linked,
           orientation='top',
           show_leaf_counts=True)

plt.show()

Z = linkage(mat, 'ward')
Z

# Extracting the useful features from the dataset
X = np.array([[1, 1], [1.5, 2], [3, 4], [4, 3], [2, 2.5], [5, 5], [7, 7], [9, 8], [8, 7], [7.5, 6.5]])

plt.scatter(X[:,0],X[:,1])

plt.show()

linked = linkage(X, 'single')
dendrogram(linked,
           orientation='top',
           show_leaf_counts=True)

plt.show()

Z = linkage(X, 'ward')
Z

# # Agglomerative Clustering 모델 생성 및 학습
# agg_clustering = AgglomerativeClustering(n_clusters=2)
# agg_clustering.fit(X)

# # 클러스터 할당 결과 확인
# labels = agg_clustering.labels_
# print("클러스터 할당 결과:", labels)

"""# **ETC**"""

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.figure(figsize=(10, 4))
ax = plt.subplot()

ddata = dendrogram(Z)

dcoord = np.array(ddata["dcoord"])
icoord = np.array(ddata["icoord"])
leaves = np.array(ddata["leaves"])
idx = np.argsort(dcoord[:, 2])
dcoord = dcoord[idx, :]
icoord = icoord[idx, :]
idx = np.argsort(Z[:, :2].ravel())
label_pos = icoord[:, 1:3].ravel()[idx][:8]

for i in range(8):
    imagebox = OffsetImage(images[i], cmap=plt.cm.bone_r, interpolation="bilinear", zoom=3)
    ab = AnnotationBbox(imagebox, (label_pos[i], 0),  box_alignment=(0.5, -0.1),
                        bboxprops={"edgecolor" : "none"})
    ax.add_artist(ab)

plt.show()

from sklearn.datasets import load_digits

digits = load_digits()
n_image = 8
np.random.seed(0)
idx = np.random.choice(range(len(digits.images)), n_image)
X = digits.data[idx]
images = digits.images[idx]

plt.figure(figsize=(12, 1))
for i in range(n_image):
    plt.subplot(1, n_image, i + 1)
    plt.imshow(images[i], cmap=plt.cm.bone)
    plt.grid(False)
    plt.xticks(())
    plt.yticks(())
    plt.title(i)

# Finding the optimal no.of Clusters by Dendrograms
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

# 덴드로그램 시각화
plt.title('덴드로그램')
plt.xlabel('고객')
plt.ylabel('유클리드 거리')
plt.show()

# 일관성 계수 계산
inconsistencies = sch.inconsistent(dendrogram['icoord'])

# 일관성 계수 그래프 플롯
plt.plot(range(1, len(inconsistencies) + 1), inconsistencies[:, 2])
plt.xlabel('군집 개수')
plt.ylabel('일관성 계수')
plt.show()

# Implementing the Hierarchical Clustering Algorithm with the optimal no.of Clusters
from sklearn.cluster import AgglomerativeClustering

def hierarchical_clustering(X, n_clusters):
    hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    return y_hc

n_clusters = 2
y_hc = hierarchical_clustering(X, n_clusters)

def clustering_plot(n_clusters):
    cmap = get_cmap('viridis')  # colormap 설정

    for i in range(0, n_clusters):
        color = cmap(i / n_clusters)  # 색상 동적 생성
        plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s=100, c=color, label='Cluster'+str(i+1))

clustering_plot(n_clusters)
plt.title('Hierarchical Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()