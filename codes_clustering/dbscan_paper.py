import os
from _table_generate import *
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Directory containing the input files
input_dir = '../files/momentum'
momentum_files = sorted(filename for filename in os.listdir(input_dir))

# Directory to save the output files
output_dir = '../files/Clustering/DBSCAN'

# Alpha for determining epsilon
alpha = 0.5  # Adjust this value based on your needs

for file in momentum_files:
    # Load the data
    data = read_and_preprocess_data(input_dir, file)

    data_array = data.values  # Exclude the first column (firm names)
    firm_names = data.index  # Get the first column (firm names)

    # Standardize the numerical data
    data_std = StandardScaler().fit_transform(data_array)

    # Compute MinPts
    MinPts = int(np.log(len(data_std)))

    # Compute epsilon
    nbrs = NearestNeighbors(n_neighbors=MinPts).fit(data_std)
    distances, indices = nbrs.kneighbors(data_std)
    distanceDec = sorted(distances[:, MinPts - 1], reverse=True)
    eps = np.percentile(distanceDec, alpha * 100)

    # Apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=MinPts, metric='manhattan').fit(data_std)
    labels = db.labels_

    # Add the cluster labels to the original data
    data['cluster'] = labels

    # Save the data with cluster labels
    data.to_csv(os.path.join(output_dir, 'clustered_' + file), index=False)