#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Libraries

# In[35]:


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.cluster import OPTICS, cluster_optics_dbscan


# # Dataset Construction

# In[47]:


data = pd.read_csv('C:/Users/김주환/Desktop/My files/PCA/2018-01.csv', header=None)
#data = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', header=None)

firm_lists = data[data.columns[0]].tolist()[1:]
data = data.set_index(data.columns[0])
data=data[1:]


LS=data.values

mat=LS[0:,1:]

print(data.head())


# In[49]:


print(LS)


# In[50]:


print(mat)


# In[51]:


print(firm_lists)


# # OPTICS

# In[71]:


X=mat

clust = OPTICS(min_samples=3, xi=0.1, min_cluster_size=3)
cluster_labels = clust.fit_predict(X)

# Get the unique cluster labels
unique_labels = set(cluster_labels)

# Create a dictionary to store firms in each cluster
clusters = {label: [] for label in unique_labels}

# Group firms by cluster label
for i, cluster_label in enumerate(cluster_labels):
    clusters[cluster_label].append(firm_lists[i])

# Print the clusters
for cluster_label, firms in clusters.items():
    print(f'Cluster {cluster_label}: {firms}')

    # Plot the line graph for firms in the cluster
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


# In[78]:


dat = pd.read_csv('C:/Users/김주환/Desktop/My files/work_data/2018-01.csv', index_col=0) 
# Get the unique cluster labels (excluding noise)
unique_labels = set(label for label in cluster_labels if label != -1)

# New table with firm name, mom_1, long and short index, cluster index
LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

clusters = {label: [] for label in unique_labels}
for i, label in enumerate(cluster_labels):
    if label != -1:  # Exclude noise
        clusters[label].append(firm_lists[i])

for cluster, firms in clusters.items():
    # Sort firms based on momentum_1
    firms_sorted = sorted(firms, key=lambda x: dat.loc[x, '1'])
    long_short = [0] * len(firms_sorted)
    for i in range(len(firms_sorted) // 2):
        long_short[i] = -1  # -1 to the low ones
        long_short[-i-1] = 1  # 1 to the high ones
        # 0 to middle point when there are odd numbers in a cluster

    # Add the data to the new table
    for i, firm in enumerate(firms_sorted):
        LS_table.loc[len(LS_table)] = [firm, dat.loc[firm, '1'], long_short[i], cluster+1]

LS_table.to_csv('C:/Users/김주환/Desktop/My files/OPTICS/2018-01.csv', index=False)

