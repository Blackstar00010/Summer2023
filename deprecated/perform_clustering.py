import Clustering as C
from PCA_and_ETC import *
from sklearn.metrics import silhouette_score

# Save Reversal method LS_Tables
Reversal_Save = False
if Reversal_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/Reversal'

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Reversal'
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        Do_Result_Save = C.ResultCheck(data)

        # Save LS_Table CSV File
        Do_Result_Save.reversal_table(data, output_dir, file)

# Save K_mean clutering method LS_Tables
# hyper parameter K(1,2,3,4,5,10,50,100,500,1000) should be tested manually.(paper follow)
K_mean_Save = True
if K_mean_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/K_Means_outlier'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    sil = 0
    cl = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.K_Mean = Do_Clustering.perform_kmeans(50)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.K_Mean[0])

        # Save LS_Table CSV File
        for i, cluster in enumerate(Do_Clustering.K_Mean):
            Do_Result_Save.ls_table(cluster, output_dir, file)

        silhouette_avg = silhouette_score(df_combined, Do_Clustering.K_Mean_labels)
        sil += silhouette_avg
        cl += len(sorted(list(set(Do_Clustering.K_Mean_labels))))
        print("The average silhouette score is:", silhouette_avg)
        print("Number of clusters is:", len(set(Do_Clustering.K_Mean_labels)))

    sil = sil / len(files)
    cl = cl / len(files)
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save DBSCAN clutering method LS_Tables
# hyper parameter eps percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
dbscan_Save = False
if dbscan_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/DBSCAN'

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/DBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    sil = 0
    cl = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(0.6)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.DBSCAN, output_dir, file)

        silhouette_avg = silhouette_score(df_combined, Do_Clustering.DBSCAN_labels)
        sil += silhouette_avg
        cl += len(set(Do_Clustering.DBSCAN_labels))
        print("The average silhouette score is:", silhouette_avg)
        print("Number of clusters is:", len(set(Do_Clustering.DBSCAN_labels)))

    sil = sil / len(files)
    cl = cl / len(files)
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save HDBSCAN clutering method LS_Tables
# hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.
hdbscan_Save = False
if hdbscan_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/HDBSCAN'

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/HDBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    sil = 0
    cl = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN(0.5)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.HDBSCAN, output_dir, file)

        if len(set(Do_Clustering.HDBSCAN_labels)) != 1:
            silhouette_avg = silhouette_score(df_combined, Do_Clustering.HDBSCAN_labels)
            sil += silhouette_avg
            print("The average silhouette score is:", silhouette_avg)
        cl += len(set(Do_Clustering.HDBSCAN_labels))

        print("Number of clusters is:", len(set(Do_Clustering.HDBSCAN_labels)))

    sil = sil / len(files)
    cl = cl / len(files)
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save Hirarchical Agglomerative clutering method LS_Tables
# hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
Agglormerative_Save = False
if Agglormerative_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/Hierarchical_Agglomerative'

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Hierarchical_Agglomerative'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    sil = 0
    cl = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        raw = False
        if not raw:
            df_combined = generate_PCA_Data(data)
        else:
            df_combined = data
        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.Agglomerative = Do_Clustering.perform_HA(0.5, draw_dendro=False)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Agglomerative)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.Agglomerative, output_dir, file, save=False, raw=raw)

        if len(sorted(list(set(Do_Clustering.Agglomerative_labels)))) != 1:
            silhouette_avg = silhouette_score(df_combined, Do_Clustering.Agglomerative_labels)
            sil += silhouette_avg
            print("The average silhouette score is:", silhouette_avg)
        cl += len(sorted(list(set(Do_Clustering.Agglomerative_labels))))

        print("Number of clusters is:", len(sorted(list(set(Do_Clustering.Agglomerative_labels)))))

    sil = sil / len(files)
    cl = cl / len(files)
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save GaussianMixture clutering method LS_Tables
# hyper parameter outlier probability [1, 5, 10, 15, 20] should be tested manually.
GMM_Save = False
if GMM_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/Gaussian_Mixture_Model'

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Gaussian_Mixture_Model'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    sil = 0
    cl = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.Gaussian = Do_Clustering.perform_GMM(1)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.Gaussian, output_dir, file)

        if len(sorted(list(set(Do_Clustering.Gaussian_labels)))) != 1:
            silhouette_avg = silhouette_score(df_combined, Do_Clustering.Gaussian_labels)
            sil += silhouette_avg
            print("The average silhouette score is:", silhouette_avg)
        cl += len(sorted(list(set(Do_Clustering.Gaussian_labels))))

        print("Number of clusters is:", len(sorted(list(set(Do_Clustering.Gaussian_labels)))))
    sil = sil / len(files)
    cl = cl / len(files)
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save OPTICS clutering method LS_Tables
# hyper parameter percentile of xi range(0.05, 0.09, 0.01) should be tested manually.
optics_Save = False
if optics_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/OPTICS'

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/OPTICS'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    sil = 0
    cl = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(0.7)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.OPTIC, output_dir, file)

        if len(sorted(list(set(Do_Clustering.OPTIC_labels)))) != 1:
            silhouette_avg = silhouette_score(df_combined, Do_Clustering.OPTIC_labels)
            sil += silhouette_avg
            print("The average silhouette score is:", silhouette_avg)

        cl += len(sorted(list(set(Do_Clustering.OPTIC_labels))))

        print("Number of clusters is:", len(sorted(list(set(Do_Clustering.OPTIC_labels)))))

    sil = sil / len(files)
    cl = cl / len(files)
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save Meanshift clutering method LS_Tables
# hyper parameter quantile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
meanshift_Save = False
if meanshift_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/Meanshift'

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Meanshift'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    sil = 0
    cl = 0
    sil_num = 0
    for file in files:
        print(file)
        # if (int(file[:4]) < 2017):
        #     continue

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.menshift = Do_Clustering.perform_meanshift(0.9)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.meanshift)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.meanshift, output_dir, file)

        if len(sorted(list(set(Do_Clustering.meanshift_labels)))) != 1:
            silhouette_avg = silhouette_score(df_combined, Do_Clustering.meanshift_labels)
            sil += silhouette_avg
            sil_num += 1
            print("The average silhouette score is:", silhouette_avg)

        cl += len(sorted(list(set(Do_Clustering.meanshift_labels))))

        print("Number of clusters is:", len(sorted(list(set(Do_Clustering.meanshift_labels)))))

    sil = sil / sil_num
    cl = cl / len(files)
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save BIRCH clutering method LS_Tables
# hyper parameter percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
birch_Save = False
if birch_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/HDBSCAN'

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/BIRCH'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    sil = 0
    cl = 0
    sil_num = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.BIRCH = Do_Clustering.perform_BIRCH(0.7)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.BIRCH)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.BIRCH, output_dir, file)

        if len(sorted(list(set(Do_Clustering.BIRCH_labels)))) != 1:
            silhouette_avg = silhouette_score(df_combined, Do_Clustering.BIRCH_labels)
            sil += silhouette_avg
            sil_num += 1
            print("The average silhouette score is:", silhouette_avg)

        cl += len(sorted(list(set(Do_Clustering.BIRCH_labels))))

        print("Number of clusters is:", len(sorted(list(set(Do_Clustering.BIRCH_labels)))))

    sil = sil / sil_num
    cl = cl / len(files)
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')


    # def outliers(self, k_value: int, alpha: float = 0.5):
    #     """
    #     :param k_value: the number of clusters
    #     :param alpha: the rate at which outliers are filtered
    #     :return: 2D list
    #     """
    #     self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)
    #     # Exclude the first column (firm names) & Exclude MOM_1
    #
    #     kmeans = BisectingKMeans(n_clusters=k_value, init='k-means++', n_init=10, max_iter=500,
    #                              algorithm='elkan', bisecting_strategy='largest_cluster').fit(self.PCA_Data)
    #
    #     cluster_labels = kmeans.labels_  # Label of each point(ndarray of shape)
    #
    #     self.test = kmeans
    #     self.K_Mean_labels = cluster_labels
    #
    #     # Calculate outlier
    #     distance = kmeans.fit_transform(self.PCA_Data)  # Distance btw K centroid about all each points
    #     main_distance = np.min(distance, axis=1)  # Distance btw own K centroid about all each points
    #
    #     main_distance_clustering = [[] for _ in range(k_value)]
    #
    #     for cluster_num in range(k_value):
    #         distance_clustering = distance[cluster_labels == cluster_num]
    #         for i in range(len(distance_clustering)):
    #             main_distance_clustering[cluster_num].append(distance_clustering[i][cluster_num])
    #
    #     # max distance in own cluster
    #     for i, cluster in enumerate(main_distance_clustering):
    #         if not cluster:
    #             continue
    #         main_distance_clustering[i] = np.max(cluster)
    #
    #     max_main_distance_clustering = main_distance_clustering
    #
    #     clusters = [[] for _ in range(k_value)]  # Cluster별 distance 분류
    #     clusters_index = [[] for _ in range(k_value)]  # Cluster별 index 분류
    #
    #     for i, cluster_num in enumerate(cluster_labels):
    #         clusters[cluster_num].append(main_distance[i])
    #         clusters_index[cluster_num].append(self.index[i])
    #
    #     outliers = [[] for _ in range(k_value)]  # Cluster별 outliers' distance 분류
    #     for i, cluster in enumerate(clusters):
    #         if max_main_distance_clustering[i] == 0:
    #             continue
    #         for j, distance in enumerate(cluster):  # distance = 자기가 속한 클러스터 내에서 중심과의 거리, cluster별로 계산해야 함.
    #             if distance / max_main_distance_clustering[i] >= alpha:
    #                 # distance / 소속 cluster 점들 중 중심과 가장 먼 점의 거리 비율이 alpha 이상이면 outlier 분류
    #                 outliers[i].append(distance)
    #
    #     outliers_index = []  # Cluster별 outliers's index 분류
    #     for i, cluster in enumerate(clusters):
    #         if not outliers[i]:
    #             continue
    #         for k, outlier_dis in enumerate(outliers[i]):
    #             for j, distance in enumerate(cluster):
    #                 if outlier_dis == distance:
    #                     outliers_index.append(clusters_index[i][j])
    #                 else:
    #                     continue
    #
    #     outliers_index = list(set(outliers_index))
    #
    #     # a에 있는 값을 b에서 빼기
    #     for value in outliers_index:
    #         for row in clusters_index:
    #             if value in row:
    #                 row.remove(value)
    #
    #     clusters_index = [sublist for sublist in clusters_index if sublist]  # 빈 리스트 제거
    #
    #     # 1차원 리스트로 전환된 outlier를 cluster 맨앞에 저장.
    #     clusters_index.insert(0, outliers_index)
    #
    #     return clusters_index

    # def perform_kmeans(self, k_value: int, alpha: float = 0.5):
    #     """
    #     :param k_value: k value to be tested
    #     :param alpha: the rate at which outliers are filtered
    #     :return: 3D list
    #     """
    #     clusters_k = []
    #     n_sample = self.PCA_Data.shape[0]  # number of values in the file
    #     # Skip if the number of values are less than k
    #     if n_sample <= k_value:
    #         k_value = n_sample
    #     clusters = self.outliers(k_value, alpha)
    #     clusters_k.append(clusters)
    #
    #     self.K_Mean = clusters_k
    #     return self.K_Mean
    #
    # ls_origin = False
    # if ls_origin:
    #     all_diffs = []
    #     for cluster_num, firms in enumerate(cluster):
    #         firms_sorted = sorted(firms, key=lambda x: self.PCA_Data.loc[x, mom1_col_name])
    #
    #         for i in range(len(firms_sorted) // 2):
    #             mom_diff = abs(self.PCA_Data.loc[firms_sorted[i], mom1_col_name] - self.PCA_Data.loc[
    #                 firms_sorted[-i - 1], mom1_col_name])
    #             all_diffs.append(mom_diff)
    #
    #     std_dev = np.std(all_diffs)
    #
    #     for cluster_num, firms in enumerate(cluster):
    #         firms_sorted = sorted(firms, key=lambda x: self.PCA_Data.loc[x, mom1_col_name])
    #         long_short = [0] * len(firms_sorted)
    #
    #         for i in range(len(firms_sorted) // 2):
    #             # Only assign long-short indices if the mom1 difference is greater than the standard deviation
    #             if abs(self.PCA_Data.loc[firms_sorted[i], mom1_col_name] - self.PCA_Data.loc[
    #                 firms_sorted[-i - 1], mom1_col_name]) > std_dev:
    #                 long_short[i] = 1  # 1 to the low ones
    #                 long_short[-i - 1] = -1  # -1 to the high ones
    #                 # 0 to middle point when there are odd numbers in a cluster
    #
    #         # Outlier cluster를 빼지 않는 대신 LS_Value를 0으로
    #
    #         if cluster_num == 0:
    #             long_short = [0] * len(firms_sorted)
    #
    #         # Add the data to the new table
    #         for i, firm in enumerate(firms_sorted):
    #             LS_table.loc[len(LS_table)] = [firm, self.PCA_Data.loc[firm, mom1_col_name], long_short[i],
    #                                            cluster_num]
    #
    #     LS_table.sort_values('Cluster Index', inplace=True)
    #     # Save the output to a CSV file in the output directory
    #     if save:
    #         LS_table.to_csv(os.path.join(output_dir, file), index=False)
    #         print(f'Exported to {output_dir}!')

def find_cointegrated_pairs_deprecated(data: pd.DataFrame):
        """
        Deprecated
        :param data:
        :return:
        """
        data = data.iloc[1:, :]
        invest_list = []

        pairs = list(combinations(data.columns, 2))  # 모든 회사 조합
        print(len(pairs))
        pairs_len = 1

        count_p = 0
        count_s = 0
        count_n = 0

        while len(pairs) != pairs_len:
            pairs_len = len(pairs)

            for i, pair in enumerate(pairs):
                pvalue = cointegrate(data, pair[0], pair[1])

                if pvalue > 0.01:
                    continue

                else:
                    count_p += 1
                    spread = data[pair[0]] - data[pair[1]]
                    adf_result = sm.tsa.adfuller(spread)
                    kpss_result = kpss(spread)

                    if adf_result[1] > 0.05 or kpss_result[1] < 0.05:
                        continue

                    else:
                        count_s += 1
                        mean_spread = spread.mean()
                        std_spread = spread.std()
                        z_score = (spread - mean_spread) / std_spread

                        spread_value = float(z_score[0])

                        if abs(spread_value) <= 2:
                            continue


                        elif spread_value < -2:
                            pair = (pair[1], pair[0])
                            # pair = (pair[1], pair[0], pvalue, adf_result[1], kpss_result[1], spread_value)
                            invest_list.append(pair)
                            pairs = [p for p in pairs if all(item not in pair for item in p)]
                            count_n += 1

                            break

                        elif spread_value > 2:
                            pair = (pair[0], pair[1])
                            # pair = (pair[0], pair[1], pvalue, adf_result[1], kpss_result[1], spread_value)
                            invest_list.append(pair)
                            pairs = [p for p in pairs if all(item not in pair for item in p)]
                            count_n += 1

                            break

            print(len(pairs))
            print(len(invest_list))

        print(f'pvalue {count_p}')
        print(f'stationary {count_s}')
        print(f'final {count_n}')

        return invest_list