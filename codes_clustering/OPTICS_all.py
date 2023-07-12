#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import csv
import urllib.parse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.cluster import OPTICS, cluster_optics_dbscan


# Specify the directories for raw data and PCA results
#raw_data_dir = 'C:/Users/김주환/Desktop/My files/PCA'
#pca_output_dir = 'C:/Users/김주환/Desktop/My files/OPTICS'
raw_data_dir = 'C:/Users/IE/Desktop/My files/PCA'
pca_output_dir = 'C:/Users/IE/Desktop/My files/OPTICS'


# Get a list of all CSV files in the raw data directory
csv_files = [file for file in os.listdir(raw_data_dir) if file.endswith('.csv')]


# Loop through each CSV file
for file in csv_files:

    csv_path = os.path.join(raw_data_dir, file)
    data = pd.read_csv(csv_path, header=None)


    firm_lists = data[data.columns[0]].tolist()[1:]
    data = data.set_index(data.columns[0])
    data=data[1:]


    LS=data.values
    mat=LS[0:,1:]



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
        #print(f'Cluster {cluster_label}: {firms}')

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



    dat = pd.read_csv(csv_path, index_col=0) 
    # Get the unique cluster labels (excluding noise)
    unique_labels = set(label for label in cluster_labels if label != -1)

    # Save the PCA results as CSV in the PCA output directory
    output_file = os.path.join(pca_output_dir, file)
    
    # New table with firm name, mom_1, long and short index, cluster index
    LS_table = pd.DataFrame(columns=['Firm', 'Mom1', 'LS', 'Cluster'])

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


    LS_table.to_csv(output_file, index=False)
    
    # Print the download link and file path for the saved CSV file
    download_link = urllib.parse.quote(output_file)
    file_path = os.path.abspath(output_file)
    print(f"Download link: {download_link}")
    print(f"File path: {file_path}")

