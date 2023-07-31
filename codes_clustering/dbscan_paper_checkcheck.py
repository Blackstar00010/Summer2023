from sklearn.cluster import DBSCAN


def successful_params(data_array, eps_values):
    successful_params = []
    # TODO minsamples = 회사 개수

    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
        cluster_labels = dbscan.fit_predict(data_array)

        unique_labels = set(label for label in cluster_labels if label != -1)

        if len(unique_labels) >= 2:
            successful_params.append([eps, min_samples])

    return successful_params
