from t_SNE import *
from _Cluster_Plot import *
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from dbscan_checkcheck import successful_params

input_dir = '../files/PCA/PCA(1-48)'
file = '1993-01.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)
firm_names = data.index

eps_values = np.linspace(0.01, 3., 300)
min_samples_values = range(2, 20)

successful_params = successful_params(data_array, eps_values, min_samples_values)

def perform_DBSCAN2(data_array, successful_params):

    lab = True
    if lab:
        output = []

        for i, example in enumerate(successful_params):
            for j in range(1):
                eps = example[j]
                ms = example[j + 1]

                labels = DBSCAN(min_samples=ms, eps=eps, metric='manhattan').fit(data_array).labels_

                score = silhouette_score(data_array, labels)
                # silhouette score 높을 수록 클러스터링 잘 된것. from -1 to 1
                output.append([ms, eps, score])

        min_samples, eps, score = sorted(output, key=lambda x: x[-1])[-1]


        labels = DBSCAN(min_samples=min_samples, eps=eps, metric='manhattan').fit(data_array).labels_

    return labels


if __name__ == "__main__":
    labels = perform_DBSCAN2(data_array, successful_params)

    # Get the unique cluster labels
    unique_labels = sorted(list(set(labels)))

    clust = [[] for _ in unique_labels]
    for i, cluster_label in enumerate(labels):
       clust[unique_labels.index(cluster_label)].append(data.index[i])

    # 3. Print and plot the clusters
    for i, firms in enumerate(clust):
        plot_clusters(unique_labels[i], firms, data.index, data_array)  # Use the imported function

    t_SNE(data_array, labels)
