from t_SNE import *
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from dbscan_checkcheck import successful_params

input_dir = '../files/PCA/PCA(1-48)'
file = '2022-12.csv'
data = read_and_preprocess_data(input_dir, file)
data_array = data.values[:, 1:].astype(float)
firm_names = data.index

eps_values = np.linspace(0.01, 2., 200)
min_samples_values = range(2, 20)

successful_params = successful_params(data, eps_values, min_samples_values)


def perform_DBSCAN2(data_array, successful_params):
    lab = True
    if lab:
        # list into dataframe
        data_frame = pd.DataFrame(data_array)

        # StandardScaler to calculate faster
        st = StandardScaler()
        stdDf = pd.DataFrame(st.fit_transform(data_frame), columns=data_frame.columns)

        output = []

        for eps in [i[0] for i in successful_params]:
            for ms in [i[1] for i in successful_params]:
                labels = DBSCAN(min_samples=ms, eps=eps, metric='manhattan').fit(stdDf).labels_

                score = silhouette_score(stdDf, labels)
                # silhouette score 높을 수록 클러스터링 잘 된것. from -1 to 1
                output.append([ms, eps, score])

        min_samples, eps, score = sorted(output, key=lambda x: x[-1])[-1]

        labels = DBSCAN(min_samples=min_samples, eps=eps, metric='manhattan').fit(stdDf).labels_
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')

    return output, labels, dbscan


if __name__ == "__main__":
    output, labels, dbscan = perform_DBSCAN2(data_array, successful_params)
    print(labels)

    # # Get the unique cluster labels
    # unique_labels = sorted(list(set(cluster_labels)))

    # clust = [[] for _ in unique_labels]
    # for i, cluster_label in enumerate(cluster_labels):
    #    clust[unique_labels.index(cluster_label)].append(data.index[i])
    #
    # # 3. Print and plot the clusters
    # for i, firms in enumerate(clust):
    #     plot_clusters(unique_labels[i], firms, data.index, data_array)  # Use the imported function
    #
    # t_SNE(data_array, dbscan)
