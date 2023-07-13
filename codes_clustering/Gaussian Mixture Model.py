#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Libraries

# In[112]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from Cluster_Plot import plot_clusters


# # Dataset Construction

# In[113]:


#data = pd.read_csv('C:/Users/김주환/Desktop/My files/PCA/2018-01.csv', header=None, index_col=[0])
data = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', header=None, index_col=[0])
data=data[1:]
LS=data.values
mata=LS[0:,1:]


# In[114]:


mata = mata.astype(float)
LS = LS.astype(float)
print('Mom1+PCA')
print(LS)
print('Only PCA')
print(mata)


# In[115]:


DEBUG = True
def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k, allow_singular=True)
    return norm.pdf(Y)


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

# In[116]:


DEBUG = True
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

class_indices = []
for i in range(K):
    indices = [data.index[index] for index, c in enumerate(category) if c == i][0:]
    class_indices.append(indices)

for i, indices in enumerate(class_indices):
    f"Class {i+1} indices: {indices}"
    
class_indices_dict = {}
for i, indices in enumerate(class_indices):
    class_name = f"Class {i+1}"
    class_indices_dict[class_name] = indices
    
cluster_elements = {i: [] for i in range(1, K+1)}
for i in range(N):
    cluster = category[i]
    index = data.index[i]
    value = LS[i, 0]  # 첫 번째 열의 값
    cluster_elements[cluster+1].append((index, value))
    
    
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


# In[117]:


df = pd.DataFrame(columns=['Firm', 'Value', 'Rank Value', 'Cluster'])
for cluster, elements in cluster_elements.items():
    elements.sort(key=lambda x: x[1], reverse=True)
    num_elements = len(elements)
    median_index = num_elements // 2
    median_value = elements[median_index][1]
    for rank, (index, value) in enumerate(elements):
        if value > median_value:
            rank_value = 1
        elif value < median_value:
            rank_value = -1
        else:
            rank_value = 0
        df = pd.concat([df, pd.DataFrame({'Firm': [index], 'Value': [value], 'Rank Value': [rank_value], 'Cluster': [cluster]})])
df


# In[118]:


# import os
# import csv
# import urllib.parse
# #output_file = 'C:/Users/김주환/Desktop/My files/Gaussian_Mixture_model/2018-01.csv'
# output_file = 'C:/Users/IE/Desktop/My files/Gaussian_Mixture_model/2018-01.csv'
# df.to_csv(output_file, index=False)
# download_link = urllib.parse.quote(output_file)
# file_path = os.path.abspath(output_file)
# print(f"Download link: {download_link}")
# print(f"File path: {file_path}")


# In[119]:


import pandas as pd
from sklearn.cluster import KMeans
# Read data from CSV file
data = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', index_col = 0)
firm_names = data.index  # Get the first column (firm names)
data_array = data.values[:,1:]  # Exclude the first column (firm names)

# Define the number of clusters k
k_values = [4]

# Print the clusters for each k value & plot the clusters
for i, clusters in enumerate(class_indices):
    print(f'Clusters for k = {k_values[i]}')    
    for j, firms in enumerate(class_indices):
        plot_clusters(j, firms, firm_names, mata)  # Use the imported function

