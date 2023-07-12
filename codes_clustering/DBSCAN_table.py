import os
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

# Directory containing the input files
input_dir = '../files/momentum'
momentum = sorted(filename for filename in os.listdir(input_dir))

# Directory to save the output files
output_dir = '../files/Clustering/DBSCAN'

for file in momentum:
    # Read data from CSV file
    data = pd.read_csv(os.path.join(input_dir, file), index_col=0)

    # Replace infinities with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    data.dropna(inplace=True)

    data_array = data.values  # Get the data values
    firm_names = data.index  # Get the firm names

    # Define DBSCAN parameters
    eps = 0.6805  # Maximum distance between two samples to be considered as neighbors
    min_samples = 2  # Minimum number of samples in a neighborhood for a point to be considered as a core point

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(data_array)

    # Get the unique cluster labels (excluding noise)
    unique_labels = set(label for label in cluster_labels if label != -1)

    # New table with firm name, mom_1, long and short index, cluster index
    LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

    clusters = {label: [] for label in unique_labels}
    for i, label in enumerate(cluster_labels):
        if label != -1:  # Exclude noise
            clusters[label].append(firm_names[i])

    for cluster, firms in clusters.items():
        # Sort firms based on momentum_1
        firms_sorted = sorted(firms, key=lambda x: data.loc[x, '1'])
        long_short = [0] * len(firms_sorted)
        for i in range(len(firms_sorted) // 2):
            long_short[i] = -1  # -1 to the low ones
            long_short[-i - 1] = 1  # 1 to the high ones
            # 0 to middle point when there are odd numbers in a cluster

        # Add the data to the new table
        for i, firm in enumerate(firms_sorted):
            LS_table.loc[len(LS_table)] = [firm, data.loc[firm, '1'], long_short[i], cluster + 1]

    # Save the output to a CSV file in the output directory
    LS_table.to_csv(os.path.join(output_dir, file), index=False)