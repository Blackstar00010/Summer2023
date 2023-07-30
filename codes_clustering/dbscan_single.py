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

eps_values = np.linspace(0.01, 2., 100)
min_samples_values = range(2, 20)

successful_params = successful_params(data, eps_values, min_samples_values)

# # Define DBSCAN parameters
# eps = 1.9651  # Maximum distance between two samples to be considered as neighbors
# min_samples = 3  # Minimum number of samples in a neighborhood for a point to be considered as a core point
#
#
# def perform_DBSCAN(data_array, eps, min_samples):
#     # Perform DBSCAN clustering
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
#     cluster_labels = dbscan.fit_predict(data_array)
#
#     return cluster_labels, dbscan


def perform_DBSCAN2(data_array, successful_params):
    lab = True
    if lab:
        # list into dataframe
        data_frame = pd.DataFrame(data_array)

        # StandardScaler to calculate faster
        st = StandardScaler()
        stdDf = pd.DataFrame(st.fit_transform(data_frame), columns=data_frame.columns)

        # # NearestNeighbors 모델 생성
        # # 각 점마다 10개의 가장 가까운 점 계산.
        # nn_model = NearestNeighbors(n_neighbors=20, metric='manhattan').fit(stdDf)

        # # 각 데이터 포인트에 대한 최근접 이웃 인덱스와 거리 작은 순 정렬.
        # distances, indices = nn_model.kneighbors(stdDf)
        # distances = np.sort(distances, axis=0)
        # distances = distances[:, 1]


        eps_values = np.linspace(0.01, 4., 101)
        min_samples = range(2, 16)
        # returns array of ranging from 0.05 to 0.13 with step of 0.01

        output = []

        for eps in [i[0] for i in successful_params]:
            for ms in [i[1] for i in successful_params]:
                labels = DBSCAN(min_samples=ms, eps=eps, metric='manhattan').fit(stdDf).labels_

                if len(Counter(labels)) == 1:
                    continue

                score = silhouette_score(stdDf, labels)
                #silhouette score 높을 수록 클러스터링 잘 된것. from -1 to 1
                output.append((ms, eps, score))

        min_samples, eps, score = sorted(output, key=lambda x: x[-1])[-1]
        print(f"Best silhouette_score: {score}")
        print(f"min_samples: {min_samples}")
        print(f"eps: {eps}")

        labels = DBSCAN(min_samples=min_samples, eps=eps, metric='manhattan').fit(stdDf).labels_
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
        clusters = len(Counter(labels))
        print(f"Number of clusters: {clusters}")
        print(f"Number of outliers: {Counter(labels)[-1]}")
        print(f"Silhouette_score: {silhouette_score(stdDf, labels)}")
    return output, labels, dbscan

if __name__ == "__main__":
    # cluster_labels, dbscan = perform_DBSCAN(data_array, eps, min_samples)
    output, labels, dbscan = perform_DBSCAN2(data_array)
    print(output)

    # # Get the unique cluster labels
    # unique_labels = sorted(list(set(cluster_labels)))
    #
    # clust = [[] for _ in unique_labels]
    # for i, cluster_label in enumerate(cluster_labels):
    #     clust[unique_labels.index(cluster_label)].append(data.index[i])
    #
    # # 3. Print and plot the clusters
    # for i, firms in enumerate(clust):
    #     plot_clusters(unique_labels[i], firms, data.index, data_array)  # Use the imported function
    #
    # t_SNE(data_array, dbscan)
