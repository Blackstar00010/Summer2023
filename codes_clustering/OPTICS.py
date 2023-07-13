#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Libraries

# In[10]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import OPTICS, cluster_optics_dbscan


# # Dataset Construction

# In[11]:


#data = pd.read_csv('C:/Users/김주환/Desktop/My files/PCA/2018-01.csv', header=None)
data = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', header=None)

firm_lists = data[data.columns[0]].tolist()[1:]
data = data.set_index(data.columns[0])
data=data[1:]
LS=data.values
mat=LS[0:,1:]
mat = mat.astype(float)
LS = LS.astype(float)
print('Mom1+PCA')
print(LS)
print('Only PCA')
print(mat)


# # OPTICS

# In[12]:


#OPTICS 알고리즘 구현
# xi = 거리, min_samples = 포함할 최소 데이터 수, min_cluster_size는 생성될 최소 군집 수
clust = OPTICS(min_samples=3, xi=0.1, min_cluster_size=3)
# 구분된 군집에 cluster_label부여하여 딕셔너리 형태로 저장.
cluster_labels = clust.fit_predict(mat)
unique_labels = set(cluster_labels)
clusters = {label: [] for label in unique_labels}
#Firm_lists의 index를 firm_index에 저장.
for i, cluster_label in enumerate(cluster_labels):
    clusters[cluster_label].append(firm_lists[i])
for cluster_label, firms in clusters.items():
    f'Cluster {cluster_label}: {firms}'
    for firm in firms:
        firm_index = list(firm_lists).index(firm)
        firm_data = mat[firm_index]
        plt.plot(range(1, len(firm_data) + 1), firm_data, label=firm)
    plt.xlabel('Characteristics')
    plt.ylabel('Data Value')
    plt.title(f'Cluster {cluster_label}')
    # List the firm names on the side of the graph
    if len(firms) <= 10:
        plt.legend(loc='center right')
    else:
        plt.legend(loc='center right', title=f'Total Firms: {len(firms)}', labels=firms[:10] + ['...'])

    plt.show()

    print()


# In[13]:


#Result CSV 구현 및 생성.
dat = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', index_col=0) 
unique_labels = set(label for label in cluster_labels if label != -1)
LS_table = pd.DataFrame(columns=['Firm', 'Mom1', 'LS', 'Cluster'])
clusters = {label: [] for label in unique_labels}
for i, label in enumerate(cluster_labels):
    if label != -1:
        clusters[label].append(firm_lists[i])
for cluster, firms in clusters.items():
    firms_sorted = sorted(firms, key=lambda x: dat.loc[x, '1'])
    long_short = [0] * len(firms_sorted)
    for i in range(len(firms_sorted) // 2):
        long_short[i] = -1  
        long_short[-i-1] = 1  
    for i, firm in enumerate(firms_sorted):
        LS_table.loc[len(LS_table)] = [firm, dat.loc[firm, '1'], long_short[i], cluster+1]

LS_table


# In[14]:


# import os
# import csv
# import urllib.parse
# #output_file = 'C:/Users/김주환/Desktop/My files/Gaussian_Mixture_model/2018-01.csv'
# output_file = 'C:/Users/IE/Desktop/My files/Gaussian_Mixture_model/2018-01.csv'
# LS_table.to_csv(output_file, index=False)
# download_link = urllib.parse.quote(output_file)
# file_path = os.path.abspath(output_file)
# print(f"Download link: {download_link}")
# print(f"File path: {file_path}")

