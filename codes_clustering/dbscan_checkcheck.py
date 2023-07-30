from _table_generate import *
from sklearn.cluster import DBSCAN
import numpy as np


def successful_params(data, eps_values, min_samples_values):
    successful_params = []

    data_array = data.values

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
            cluster_labels = dbscan.fit_predict(data_array)

            unique_labels = set(label for label in cluster_labels if label != -1)

            if len(unique_labels) >= 2:
                successful_params.append([eps, min_samples])

    return successful_params
