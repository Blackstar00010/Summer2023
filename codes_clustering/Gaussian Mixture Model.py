#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Libraries

# In[265]:


import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# # Dataset Construction

# In[266]:


# Specify the file path
data = pd.read_csv('C:/Users/김주환/Desktop/My files/PCA/2018-01.csv', header=None, index_col=[0])
#data = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', header=None, index_col=[0])

# firms_list = data[data.columns[0]].tolist()[1:]
# data = data.set_index(data.columns[0])
data=data[1:]


LS=data.values

mata=LS[0:,1:]

print(data.index)


# In[267]:


mata = mata.astype(float)
LS = LS.astype(float)

print(LS)


# In[268]:


print(mata)


# In[269]:


# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

DEBUG = True


def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)



def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k, allow_singular=True)
    return norm.pdf(Y)


#원본
def getExpectation(Y, mu, cov, alpha):
    N = Y.shape[0]
    K = alpha.shape[0]

    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"


    gamma = np.mat(np.zeros((N, K)))
    prob = np.zeros((N, K))
    
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    
    prob = np.mat(prob)


    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
        
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    
    return gamma


# #GPT
# def getExpectation(Y, mu, cov, alpha):
#     N = Y.shape[0]
#     K = alpha.shape[0]

#     assert N > 1, "There must be more than one sample!"
#     assert K > 1, "There must be more than one gaussian model!"


#     gamma = np.mat(np.zeros((N, K)))
#     prob = np.zeros((N, K))
    
#     for k in range(K):
#         prob[:, k] = phi(Y, mu[k], cov[k])
    
#     prob = np.mat(prob)


#     for k in range(K):
#         norm = multivariate_normal(mean=mu[k], cov=cov[k], allow_singular=True)
#         gamma[:, k] = alpha[k] * norm.pdf(Y)
        
#     for i in range(N):
#         gamma[i, :] /= np.sum(gamma[i, :])
    
#     return gamma





def maximize(Y, gamma):

    N, D = Y.shape

    K = gamma.shape[1]


    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)


    for k in range(K):

        Nk = np.sum(gamma[:, k])

        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk

        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)

        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha


def scale_data(Y):

    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    debug("Data scaled.")
    return Y



def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    debug("Parameters initialized.")
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha



def GMM_EM(Y, K, times):
    Y = scale_data(Y)
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    debug("{sep} Result {sep}".format(sep="-" * 20))
    debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


# # GMM

# In[270]:


#np.savetxt("mat.data", mat)

plt.plot(mata[:20, 0], mata[:20, 1], "bo")
plt.plot(mata[20:, 0], mata[20:, 1], "rs")
plt.show()


# In[271]:


# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------
# allow_singular=True

DEBUG = True


#Y = np.loadtxt("mata.data")
Y=mata
matY = np.matrix(Y, copy=True)


K = 4


mu, cov, alpha = GMM_EM(matY, K, 100)


N = Y.shape[0]

gamma = getExpectation(matY, mu, cov, alpha)

category = gamma.argmax(axis=1).flatten().tolist()[0]


class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
class3 = np.array([Y[i] for i in range(N) if category[i] == 2])
class4 = np.array([Y[i] for i in range(N) if category[i] == 3])

plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
plt.plot(class3[:, 0], class3[:, 1], 'go', label="class3")
plt.plot(class4[:, 0], class4[:, 1], 'cd', label="class4")

plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.show()

print(len(class1)) 
print(len(class2))
print(len(class3))
print(len(class4))


# In[272]:


print(matY)
print(len(matY))
print(len(matY[1]))

print(cov)
print(len(cov))
print(len(cov[1]))


# In[273]:


class_indices = []
for i in range(K):
    indices = [data.index[index] for index, c in enumerate(category) if c == i][0:]
    class_indices.append(indices)

for i, indices in enumerate(class_indices):
    print(f"Class {i+1} indices: {indices}")


# In[274]:


class_indices_dict = {}
for i, indices in enumerate(class_indices):
    class_name = f"Class {i+1}"
    class_indices_dict[class_name] = indices

print(class_indices_dict)


# In[275]:


import os
import csv
import urllib.parse

print(LS[0,0])
cluster_elements = {i: [] for i in range(1, K+1)}
print(cluster_elements)

for i in range(N):
    cluster = category[i]
    index = data.index[i]
    value = LS[i, 0]  # 첫 번째 열의 값
    cluster_elements[cluster+1].append((index, value))


# In[276]:


# 각 클러스터 내부의 첫 번째 열 값에 따라 내림차순 정렬
for cluster, elements in cluster_elements.items():
    elements.sort(key=lambda x: x[1], reverse=True)

    num_elements = len(elements)
    if num_elements % 2 == 1:
        median_index = num_elements // 2  # 중간값의 인덱스
        median_value = elements[median_index][1]  # 중간값의 값
        for rank, (index, value) in enumerate(elements):
            if value > median_value:
                rank_value = 1  # 상위 rank에는 1 할당
            elif value < median_value:
                rank_value = -1  # 하위 rank에는 -1 할당
            else:
                rank_value = 0  # 중간값에는 0 할당
            print(f"Firm: {index}, Value: {value}, Rank Value: {rank_value}, Cluster: {cluster}")
    else:
        upper_half = num_elements // 2
        lower_half = num_elements // 2
        for rank, (index, value) in enumerate(elements):
            if rank < upper_half:
                rank_value = 1  # 상위 rank에는 1 할당
            elif rank >= num_elements - lower_half:
                rank_value = -1  # 하위 rank에는 -1 할당
            else:
                rank_value = 0  # 중간값에는 0 할당
            print(f"Firm: {index}, Value: {value}, Rank Value: {rank_value}, Cluster: {cluster}")
            


# In[277]:


import pandas as pd

# 출력값을 저장할 데이터 프레임
df = pd.DataFrame(columns=['Firm', 'Value', 'Rank Value', 'Cluster'])

# 각 클러스터 내부의 첫 번째 열 값에 따라 내림차순 정렬
for cluster, elements in cluster_elements.items():
    elements.sort(key=lambda x: x[1], reverse=True)

    num_elements = len(elements)
    if num_elements % 2 == 1:
        median_index = num_elements // 2  # 중간값의 인덱스
        median_value = elements[median_index][1]  # 중간값의 값
        for rank, (index, value) in enumerate(elements):
            if value > median_value:
                rank_value = 1  # 상위 rank에는 1 할당
            elif value < median_value:
                rank_value = -1  # 하위 rank에는 -1 할당
            else:
                rank_value = 0  # 중간값에는 0 할당
            # 데이터 프레임에 출력값 추가
            df = pd.concat([df, pd.DataFrame({'Firm': [index], 'Value': [value], 'Rank Value': [rank_value], 'Cluster': [cluster]})])
    else:
        upper_half = num_elements // 2
        lower_half = num_elements // 2
        for rank, (index, value) in enumerate(elements):
            if rank < upper_half:
                rank_value = 1  # 상위 rank에는 1 할당
            elif rank >= num_elements - lower_half:
                rank_value = -1  # 하위 rank에는 -1 할당
            else:
                rank_value = 0  # 중간값에는 0 할당
            # 데이터 프레임에 출력값 추가
            df = pd.concat([df, pd.DataFrame({'Firm': [index], 'Value': [value], 'Rank Value': [rank_value], 'Cluster': [cluster]})])

# 데이터 프레임 출력
print(df)


# In[278]:


# 첫 번째 열과 인덱스를 일치하게 정렬
df_sorted = df.set_index('Firm').loc[data.index].reset_index()


# In[279]:


print(df_sorted)


# In[280]:


import os
import csv
import urllib.parse

# 파일 경로와 이름 정의
#output_file = 'C:/Users/IE/Desktop/My files/Gaussian_Mixture_model/2018-01.csv'
output_file = 'C:/Users/김주환/Desktop/My files/Gaussian_Mixture_model/2018-01.csv'


# 데이터프레임을 CSV 파일로 저장
df_sorted.to_csv(output_file, index=False)

# 다운로드 링크와 파일 경로 출력
download_link = urllib.parse.quote(output_file)
file_path = os.path.abspath(output_file)
print(f"Download link: {download_link}")
print(f"File path: {file_path}")

