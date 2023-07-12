#!/usr/bin/env python
# coding: utf-8

# # **Import Necessary Libraries**

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
get_ipython().run_line_magic('matplotlib', 'inline')


# # **Dataset Construction**

# In[21]:


data = pd.read_csv('C:/Users/김주환/Desktop/My files/PCA/2018-01.csv', header=None)
#data = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', header=None)

firms_list = data[data.columns[0]].tolist()[1:]
data = data.set_index(data.columns[0])
data=data[1:]


LS=data.values

mat=LS[0:,1:]

print(data.head())


# In[22]:


print(len(firms_list))


# In[23]:


print(LS)


# In[24]:


print(mat)


# In[25]:


print(mat.shape)


# In[26]:


print(LS.shape)


# # **Hierarchical Agglomerative Clustering**

# In[27]:


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


# In[28]:


len(clusters)


# In[29]:


#Cluster별 분류
list_dict = {}
for i in range(k):
     list_dict[i+1] = []

for i in range(0,463):
    for j in range(0,k):
        if clusters[i]==j+1:
          list(list_dict.values())[j].append('firm'+str(i+1)+': '+str(LS[0:,0][i]))

for key, value in list_dict.items():
    print(value)
    print(key)

# first_value = list_dict[1]

# print(first_value)


# In[30]:


#LS와 Rank
for key, value in list_dict.items():
    print(f"Cluster: {key}")
    sorted_data = sorted(value, key=lambda x: float(x.split(': ')[1]), reverse=True)
    length = len(sorted_data)
    for i, item in enumerate(sorted_data):
        rank = i - length // 2
        if length%2==0:
          if rank == 0:
            h=int((length)/2)+1
            t=1


          if rank > 0:
            h=rank+int((length)/2)+1
            t=1

          elif rank < 0:
            h=rank+int((length)/2)+1
            t=-1

          print(f"{item}: {t}: {h}")

        else:
          if rank == 0:
            h=rank+int((length-1)/2)+1
            t=0

          if rank > 0:
            h=rank+int((length-1)/2)+1
            t=1

          elif rank < 0:
            h=rank+int((length-1)/2)+1
            t=-1

          print(f"{item}: {t}: {h}")



# In[31]:


#행렬로 변환
result_matrix = []

for key, value in list_dict.items():
    row = [key]  # Cluster 값을 첫 번째 열에 저장
    sorted_data = sorted(value, key=lambda x: float(x.split(': ')[1]), reverse=True)
    length = len(sorted_data)
    for i, item in enumerate(sorted_data):
        rank = i - length // 2
        if length%2==0:
          if rank == 0:
            h=int((length)/2)+1
            t=1


          if rank > 0:
            h=rank+int((length)/2)+1
            t=1

          elif rank < 0:
            h=rank+int((length)/2)+1
            t=-1

          row.append(f"{item}: {t}: {h}")

        else:
          if rank == 0:
            h=rank+int((length-1)/2)+1
            t=0

          if rank > 0:
            h=rank+int((length-1)/2)+1
            t=1

          elif rank < 0:
            h=rank+int((length-1)/2)+1
            t=-1

          row.append(f"{item}: {t}: {h}")





    result_matrix.append(row)  # 행을 결과 행렬에 추가

# 결과 행렬 출력
for row in result_matrix:
    print(row)


# In[32]:


#행렬 csv로 변환
import pandas as pd

result_matrix = []

for key, value in list_dict.items():
    row = [key]  # Cluster 값을 첫 번째 열에 저장
    sorted_data = sorted(value, key=lambda x: float(x.split(': ')[1]), reverse=True)
    length = len(sorted_data)
    for i, item in enumerate(sorted_data):
        rank = i - length // 2
        if length%2==0:
          if rank == 0:
            h=int((length)/2)+1
            t=1


          if rank > 0:
            h=rank+int((length)/2)+1
            t=1

          elif rank < 0:
            h=rank+int((length)/2)+1
            t=-1

          row.append(f"{item}: {t}: {h}")

        else:
          if rank == 0:
            h=rank+int((length-1)/2)+1
            t=0

          if rank > 0:
            h=rank+int((length-1)/2)+1
            t=1

          elif rank < 0:
            h=rank+int((length-1)/2)+1
            t=-1

          row.append(f"{item}: {t}: {h}")





    result_matrix.append(row)  # 행을 결과 행렬에 추가

# 결과 행렬을 DataFrame으로 변환
df_result = pd.DataFrame(result_matrix)

# # CSV 파일로 저장
# df_result.to_csv('output.csv', index=True)


# In[33]:


# # Importing the dataset using pandas dataframe
# df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/output.csv', header=None, index_col=[0])
# df = df.drop(df.index[0])
# df


# 저장한 파일을 바로 pd.read_csv로 읽기
# df = pd.read_csv('output.csv', header=None, index_col=0)
df = df_result




# print(df.iloc[99, 0])
# print(df.iloc[1, 0])
# print(df.iloc[2, 0])
# print(df.iloc[3, 0])
# print(len(df.head(2).iloc[1].dropna()))
# df.iloc[1, 1] = str(df.iloc[1, 1]) + ':' + str(df.iloc[1, 0])
# print(df.iloc[1, 1])

#맨 뒤에 소속 Cluster 붙이기
for i in range(0,100):
  for j in range(0,len(df.head(i+1).iloc[i].dropna())-1):
    df.iloc[i, j+1] = str(df.iloc[i, j+1]) + ':' + str(df.iloc[i, 0])

df = df.iloc[:, 1:]

df

#data frame into list
result_list = []

for row in df.values:
    for item in row:
        if pd.notnull(item):  # 결측값이 아닌 경우에만 리스트에 추가
            result_list.append(item)


#회사 기준 정렬
sorted_result_list = sorted(result_list, key=lambda x: int(x.split(":")[0][4:]))
#print(sorted_result_list)

# ":" 기준 split
split_result_list = [item.split(":") for item in sorted_result_list]
#print(split_result_list)

#column name
df_sorted = pd.DataFrame(split_result_list)
df_sorted.columns=["Firm", "mom1", "LS", "Rank", "Cluster"]
print(df_sorted)

df_sorted = df_sorted.iloc[:, 1:]

# 수정된 데이터프레임 출력
print(df_sorted)

# 데이터프레임에 'firms_list' 열 추가
df_sorted.insert(0, 'firms_list', firms_list)
print(df_sorted)


# In[34]:


import os
import csv
import urllib.parse

# 파일 경로와 이름 정의
#output_file = 'C:/Users/IE/Desktop/My files/Hierarchical_Agglomerative/2018-01.csv'
output_file = 'C:/Users/김주환/Desktop/My files/Hierarchical_Agglomerative/2018-01.csv'


# 데이터프레임을 CSV 파일로 저장
df_sorted.to_csv(output_file, index=False)

# 다운로드 링크와 파일 경로 출력
download_link = urllib.parse.quote(output_file)
file_path = os.path.abspath(output_file)
print(f"Download link: {download_link}")
print(f"File path: {file_path}")


# # ETC

# In[35]:


# # Extracting the useful features from the dataset
# plt.scatter(mat[:,0],mat[:,1])

# plt.show()

# linked = linkage(mat, 'single')
# dendrogram(linked,
#            orientation='top',
#            show_leaf_counts=True)

# plt.show()

# Z = linkage(mat, 'ward')
# Z


# In[36]:


# # Extracting the useful features from the dataset
# X = np.array([[1, 1], [1.5, 2], [3, 4], [4, 3], [2, 2.5], [5, 5], [7, 7], [9, 8], [8, 7], [7.5, 6.5]])

# plt.scatter(X[:,0],X[:,1])

# plt.show()

# linked = linkage(X, 'single')
# dendrogram(linked,
#            orientation='top',
#            show_leaf_counts=True)

# plt.show()

# Z = linkage(X, 'ward')
# Z

