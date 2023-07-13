import os
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

# Prints cases that makes more than 2 clusters
# Prints the eps and min_samples value

# Define DBSCAN parameters
eps_values = np.linspace(0.01, 4., 101)  # eps values from 0.01 to 1.01
min_samples_values = range(2, 51)  # min_samples values from 2 to 50

# Directory containing the input files
input_dir = '../files/momentum'

# Get a list of all the files in the input directory
files = sorted(os.listdir(input_dir))
files = files[-12:]

# Dictionary to store the successful parameter combinations for each file
successful_params = {}

# Process each file
for file in files:
    # Read data from CSV file
    data = pd.read_csv(os.path.join(input_dir, file), index_col=0)  # Set the first column as index

    # Replace infinities with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    data.dropna(inplace=True)

    data_array = data.values  # Get the data values

    # Iterate over all combinations of eps and min_samples
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(data_array)

            # Get the unique cluster labels (excluding noise)
            unique_labels = set(label for label in cluster_labels if label != -1)

            # If there are more than two clusters, add the parameter combination to the successful_params dictionary
            if len(unique_labels) >= 2:
                if file not in successful_params:
                    successful_params[file] = []
                successful_params[file].append((eps, min_samples))

# Find the parameter combinations that are successful for all files
common_params = set(successful_params[files[0]])
for file in files[1:]:
    common_params &= set(successful_params[file])

print('Common parameter combinations:', common_params)
