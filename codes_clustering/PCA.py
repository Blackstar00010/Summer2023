#!/usr/bin/env python
# coding: utf-8

# # **Import Necessary Libraries**

# In[102]:


import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.parse
from sklearn.decomposition import PCA


# # **Dataset Construction**

# In[103]:


data = pd.read_csv('C:/Users/김주환/Desktop/My files/raw_data/2018-01.csv', header=None)
#data = pd.read_csv('C:/Users/IE/Desktop/My files/raw_data/2018-01.csv', header=None)

firms_list = data[data.columns[0]].tolist()[1:]
data = data.set_index(data.columns[0])
data=data[1:]

print(data)

mat=data.values

mat = mat.astype(float)
print(len(mat))

print(mat)


# In[104]:


plt.scatter(mat[0,:], mat[2,:])
plt.axis('equal')

plt.xticks(range(-10, 10, 5))
plt.yticks(range(-10, 10, 5))
plt.show()


# # **Principal Component Analysis, PCA**

# In[105]:


def get_pca_data(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(data)

    return pca.transform(data), pca

pca_mat, pca= get_pca_data(mat, n_components=20)
print(pca_mat.shape)


# In[106]:


def get_pd_from_pca(pca_data, cols=None):
    if cols is None:
        cols = ['pca_component_{}'.format(i+1) for i in range(pca_data.shape[1])]
    return pd.DataFrame(pca_data, columns=cols)


# In[107]:


def print_variance_ratio(pca):
    print('variance_ratio: ', pca.explained_variance_ratio_)
    print('sum of variance_ratio: ', np.sum(pca.explained_variance_ratio_))

print_variance_ratio(pca)


# In[108]:


cols = []
for i in range(1, 21):
    cols.append('pca_' + str(i))

pca_mat_pd = get_pd_from_pca(pca_mat, cols=cols)

pca_x= pca_mat_pd[cols]


# In[109]:


pca_mat
print(pca_mat)
num_components = len(pca_mat)
print("Number of components:", num_components)

#pca_mat은 mat 데이터를 주성분 축으로 변환한 결과.


# In[110]:


pca.components_
print(pca.components_)
num_components = len(pca.components_)
print("Number of components:", num_components)
#pca.components_는 주성분 축의 방향을 나타내는 벡터들.


# In[111]:


pca.mean_
print(pca.mean_)
num_components = len(pca.mean_)
print("Number of components:", num_components)


# In[112]:


pca.explained_variance_

num_explained_variance = len(pca.explained_variance_)
print("Number of explained_variance:", num_explained_variance)


# In[113]:


pca = PCA(n_components=20)
pca.fit(mat)
X_pca=pca.transform(mat)
print("original shape: ", mat.shape)
print("transformed shape: ", pca_mat.shape)


# In[114]:


get_ipython().run_line_magic('matplotlib', 'inline')

cols = []
for i in range(1, 50):
    cols.append('pca_' + str(i))

mat_pd = pd.DataFrame(mat, columns=cols)
#mat_pd['species'] = mat_pd['pca_1']
mat_pd.head(20)


# In[115]:


mat_new=pca.inverse_transform(pca_mat)
plt.scatter(mat[:,0],mat[:,1], alpha=0.2)
plt.scatter(mat_new[:,0], mat_new[:,1], alpha=0.8)
plt.axis('equal');


# # **StandardScaler**

# In[116]:


# 각 열의 평균과 표준편차 계산
mean = np.mean(mat, axis=0)
std = np.std(mat, axis=0)

# 데이터 표준화
mat_ss = (mat - mean) / std


# In[117]:


mat[0]


# In[118]:


mat_ss[0]


# In[119]:


mat_ss[0].shape
#첫번째 행을 의미 길이가 126인 벡터를 의미


# In[120]:


mat_ss[0].reshape(1,-1).shape

#첫번째 행만 따로 때서 하나의 행렬로


# In[121]:


pca.transform(mat_ss[0].reshape(1,-1))


# In[122]:


#PCA이후 data
mat_pd_pca = get_pd_from_pca(pca_mat)
mat_pd_pca.head(20)

mat_pd_pca_matrix = mat_pd_pca.values

#원본 data의 첫열
first_column = data.iloc[:, 0]
first_column_matrix= np.array(first_column).reshape(-1, 1)
first_column_matrix=first_column_matrix[0:]

# 행렬 결합
combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))

print(mat_pd_pca_matrix)
print(combined_matrix)
print(len(combined_matrix[0]))


# In[123]:


mat_pd_pca


# In[124]:


df_combined = pd.DataFrame(combined_matrix)
df_combined


# In[125]:


len(firms_list)


# In[126]:


# 파일 경로와 이름 정의
# output_file = 'C:/Users/IE/Desktop/My files/PCA/2018-01.csv'
output_file = 'C:/Users/김주환/Desktop/My files/PCA/2018-01.csv'

# 데이터프레임에 'firms_list' 열 추가
df_combined.insert(0, 'Firm', firms_list)

# 데이터프레임을 CSV 파일로 저장
df_combined.to_csv(output_file, index=False)

# 다운로드 링크와 파일 경로 출력
download_link = urllib.parse.quote(output_file)
file_path = os.path.abspath(output_file)
print(f"Download link: {download_link}")
print(f"File path: {file_path}")

