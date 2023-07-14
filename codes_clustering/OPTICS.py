# import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
from _Cluster_Plot import plot_clusters
from sklearn.cluster import OPTICS, cluster_optics_dbscan

# 1. 파일 불러오기
# data = pd.read_csv('C:/Users/김주환/Desktop/My files/PCA/2018-01.csv', header=None)
data = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', header=None)

firm_lists = data[data.columns[0]].tolist()[1:]
data = data.set_index(data.columns[0])
data = data[1:]
LS = data.values
mat = LS[0:, 1:]
mat = mat.astype(float)
LS = LS.astype(float)
# TODO: 어떻게 생겼다
print('Mom1+PCA')
print(LS)
print('Only PCA')
print(mat)

# 2. OPTICS 알고리즘 구현
# xi = 거리, min_samples = 포함할 최소 데이터 수, min_cluster_size는 생성될 최소 군집 수
clust = OPTICS(min_samples=3, xi=0.1, min_cluster_size=3)
# 구분된 군집에 cluster_label 부여하여 dict 형태로 저장. TODO: 리스트로 하시져
cluster_labels = clust.fit_predict(mat)
unique_labels = set(cluster_labels)
clusters = {label: [] for label in unique_labels}
# Firm_lists의 index를 firm_index에 저장.
for i, cluster_label in enumerate(cluster_labels):
    clusters[cluster_label].append(firm_lists[i])
for cluster_label, firms in clusters.items():
    f'Cluster {cluster_label}: {firms}'
    for firm in firms:
        firm_index = list(firm_lists).index(firm)

# 3. OPTICS 결과출력
data_array = mat
firm_names = firm_lists
unique_labels = sorted(list(set(cluster_labels)))

# Create a list to store firms in each cluster
clusters = [[] for _ in unique_labels]

# Group firms by cluster label
for i, cluster_label in enumerate(cluster_labels):
    clusters[unique_labels.index(cluster_label)].append(firm_names[i])

# Print and plot the clusters
for i, firms in enumerate(clusters):
    plot_clusters(unique_labels[i], firms, firm_names, data_array)  # Use the imported function
    print()
