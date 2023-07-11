#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Libraries

# In[45]:


import pandas as pd
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# # Dataset Construction

# In[106]:


# Specify the file path
# data = pd.read_csv('C:/Users/김주환/Desktop/My files/work data/2018-01.csv', header=None, index_col=[0])
data = pd.read_csv('C:/Users/IE/Desktop/My files/work_data/2018-01.csv', header=None, index_col=[0])

# firms_list = data[data.columns[0]].tolist()[1:]
# data = data.set_index(data.columns[0])
data=data[1:]


LS=data.values

mat=LS[0:,1:]


# In[102]:


mat = mat.astype(float)
LS = LS.astype(float)

print(LS.shape)


# In[103]:


print(mat)


# In[75]:


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
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
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

# In[76]:


# np.savetxt("mat.data", mat)

plt.plot(mat[:20, 0], mat[:20, 1], "bo")
plt.plot(mat[20:, 0], mat[20:, 1], "rs")
plt.show()


# In[78]:


# -*- coding: utf-8 -*-
# ----------------------------------------------------
# Copyright (c) 2017, Wray Zheng. All Rights Reserved.
# Distributed under the BSD License.
# ----------------------------------------------------

import matplotlib.pyplot as plt


DEBUG = True


# Y = np.loadtxt("mat.data")
Y=mat
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


# In[113]:


class_indices = []
for i in range(K):
    indices = [data.index[index] for index, c in enumerate(category) if c == i][0:]
    class_indices.append(indices)

for i, indices in enumerate(class_indices):
    print(f"Class {i+1} indices: {indices}")


# In[116]:


class_indices_dict = {}
for i, indices in enumerate(class_indices):
    class_name = f"Class {i+1}"
    class_indices_dict[class_name] = indices

print(class_indices_dict)


# In[117]:


class_indices_dict = {}
for i, indices in enumerate(class_indices):
    class_name = f"Class {i+1}"
    class_indices_dict[class_name] = indices

for class_name, indices in class_indices_dict.items():
    count = len(indices)
    print(f"{class_name}: {count} indices")


# In[ ]:


LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

for k, clusters in clusters_k.items():
    for cluster, firms in clusters.items():
        # Sort firms based on momentum_1
        firms_sorted = sorted(firms, key=lambda x: data.loc[x, '1'])
        long_short = [0] * len(firms_sorted)
        for i in range(len(firms_sorted) // 2):
            long_short[i] = -1  # 1 to the high ones
            long_short[-i-1] = 1  # -1 to the low ones
            # 0 to middle point when there are odd numbers in a cluster

        # Add the data to the new table
        for i, firm in enumerate(firms_sorted):
            LS_table.loc[len(LS_table)] = [firm, data.loc[firm, '1'], long_short[i], cluster+1]


# In[125]:


LS_table = pd.DataFrame(columns=['Firm', 'mom1', 'LS', 'Cluster'])

for cluster, firms in class_indices_dict.items():
    # Sort firms based on momentum_1
    firms_sorted = sorted(firms, key=lambda x: data.loc[x, 'Class 1'])
    long_short = [0] * len(firms_sorted)
    for i in range(len(firms_sorted) // 2):
        long_short[i] = -1  # 1 to the high ones
        long_short[-i-1] = 1  # -1 to the low ones
        # 0 to middle point when there are odd numbers in a cluster

    # Add the data to the new table
    for i, firm in enumerate(firms_sorted):
        LS_table.loc[len(LS_table)] = [firm, data.loc[firm, '1'], long_short[i], cluster]

print(LS_table)






