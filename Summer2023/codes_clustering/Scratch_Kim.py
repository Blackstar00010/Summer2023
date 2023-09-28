from PCA_and_ETC import *

# turn off warning
warnings.filterwarnings("ignore")

abnormal = True
if abnormal:
    MOM_merged_df = pd.read_csv('../files/mom1_data_combined_adj_close.csv')
    MOM_merged_df.set_index('Firm Name', inplace=True)
    MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)

    base_directory = '../files/clustering_result/'

    # Get all subdirectories in the base directory
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    file_names = []
    result_df = pd.DataFrame()

    for subdir in subdirectories:
        print(subdir)
        directory = os.path.join(base_directory, subdir)

        LS_merged_df = pd.DataFrame()

        files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))
        for file in files:
            data = pd.read_csv(os.path.join(directory, file))
            LS_merged_df = merge_LS_Table(data, LS_merged_df, file)

        LS_merged_df.index = LS_merged_df.iloc[:, 0]
        LS_merged_df.drop(columns='Firm Name', inplace=True)

        outlier_df = pd.DataFrame()

        for i in range(len(LS_merged_df.iloc[0, :]) - 1):
            col = pd.DataFrame(LS_merged_df.iloc[:, i])

            firm_df = col.index[(col.iloc[:, 0] != 0) & (col.iloc[:, 0].notna())]

            value_df = MOM_merged_df.loc[firm_df]
            value_df = value_df.iloc[:, i]
            filtered_df = value_df[(value_df >= 1) | (value_df <= -0.5)]
            filtered_df = pd.DataFrame({filtered_df.name: filtered_df})

            outlier_df = pd.concat([outlier_df, filtered_df])

        outlier_df.to_csv(f'../files/abnormal_{subdir}.csv')

        count_numeric_df = pd.DataFrame(columns=['count'])
        # 각 열을 순회하면서 숫자가 있는 칸들의 개수를 계산하여 저장
        for col in outlier_df.columns:
            count = outlier_df[col].apply(lambda x: 1 if pd.notna(x) and isinstance(x, (int, float)) else 0).sum()
            count_numeric_df.loc[col] = [count]

        count_numeric_df.T.to_csv(f'../files/abnormal_count_{subdir}.csv')

min_max = True
if min_max:
    df = pd.read_csv('../files/mom1_data_combined_adj_close.csv')

    df = df.iloc[:, 1:]
    print(df)

    print(df.max().to_string())

    result = pd.DataFrame(df.max())

    result.to_csv('../files/max_mom1.csv')

    result = pd.DataFrame(df.min())

    result.to_csv('../files/min_mom1.csv')

t_test = False
if t_test:
    from scipy import stats

    lst_mean = [0.1265, 0.1274, 0.1356, 0.1271, 0.1164, 0.1107, 0.1053, 0.1188]
    lst_std = [0.102, 0.1048, 0.1063, 0.1049, 0.1021, 0.0917, 0.1268, 0.0864]

    for i in range(8):
        mean1 = lst_mean[i]
        std_dev1 = lst_std[i]
        sample_size1 = 384

        mean2 = 0.0856
        std_dev2 = 0.0827
        sample_size2 = 384

        # t-검정 수행
        t_statistic, p_value = stats.ttest_ind_from_stats(mean1, std_dev1, sample_size1, mean2, std_dev2, sample_size2)

        # 결과 출력
        print(f"{i}t-통계량:", t_statistic)
        print(f"{i}p-값:", p_value)

traded = True
if traded:
    traded_df = pd.read_csv('../files/traded_DBSCAN.csv')
    traded_df.columns = pd.to_datetime(traded_df.columns, errors='coerce')

    traded_df.T.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('Date')
    plt.ylabel('Number of stock traded')
    plt.title('Stock Traded Per Month')
    plt.grid(axis='y')
    plt.xticks([])
    plt.legend().set_visible(False)
    plt.show()


 def perform_kmeans(self, k_value: int, alpha: float = 0.5):



        distance_to_own_centroid = [distance.euclidean(self.PCA_Data[i], kmeans.cluster_centers_[cluster_labels[i]]) for
                                    i in range(len(self.PCA_Data))]

        nbrs = NearestNeighbors(n_neighbors=3, p=2).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        nearest_neighbor_distances = distances[:, 1]

        sorted_nearest_neighbor_distances = sorted(nearest_neighbor_distances)
        epsilon = sorted_nearest_neighbor_distances[int(len(sorted_nearest_neighbor_distances) * alpha)]
        outliers = [i for i, dist in enumerate(distance_to_own_centroid) if dist < epsilon]

        clusters_indices = [[] for _ in range(k_value)]
        for i, label in enumerate(cluster_labels):
            if i in outliers:
                continue
            clusters_indices[label].append(i)

        clusters_indices.insert(0, list(outliers))

        final_cluster = [[] for _ in clusters_indices]
        for i, num in enumerate(clusters_indices):
            for j in num:
                final_cluster[i].append(self.index[j])

        final_cluster = [cluster for cluster in final_cluster if cluster]
        self.K_Mean = final_cluster

    def perform_BIRCH(self, threshold):
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        nbrs = NearestNeighbors(n_neighbors=3, p=2).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        max_d = np.percentile(avg_distances, threshold * 100)

        birch = Birch(threshold=max_d, n_clusters=None).fit(self.PCA_Data)
        cluster_labels = birch.labels_
        self.test = birch
        self.BIRCH_labels = cluster_labels

        distance_to_own_centroid = [distance.euclidean(self.PCA_Data[i], birch.cluster_centers_[cluster_labels[i]]) for
                                    i in range(len(self.PCA_Data))]

        nbrs = NearestNeighbors(n_neighbors=3, p=2).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        nearest_neighbor_distances = distances[:, 1]

        sorted_nearest_neighbor_distances = sorted(nearest_neighbor_distances)
        epsilon = sorted_nearest_neighbor_distances[int(len(sorted_nearest_neighbor_distances) * 0.5)]
        outliers = [i for i, dist in enumerate(distance_to_own_centroid) if dist < epsilon]

        unique_labels = sorted(list(set(cluster_labels)))

        clusters_indices = [[] for _ in range(len(unique_labels))]
        for i, label in enumerate(cluster_labels):
            if i in outliers:
                continue
            clusters_indices[label].append(i)

        clusters_indices.insert(0, list(outliers))

        final_cluster = [[] for _ in clusters_indices]
        for i, num in enumerate(clusters_indices):
            for j in num:
                final_cluster[i].append(self.index[j])

        final_cluster = [cluster for cluster in final_cluster if cluster]
        self.BIRCH = final_cluster




        # Get the unique cluster labels


        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])


