import time
import warnings
from itertools import combinations

import pandas as pd
from multitasking import *
import seaborn as sns
import Clustering as C
import statsmodels.api as sm
from PCA_and_ETC import *

from sklearn.datasets import load_iris
from statsmodels.tsa.stattools import coint

# turn off warning
warnings.filterwarnings("ignore")

# sample data
iris = load_iris()
iris_pd = pd.DataFrame(iris.data[:, 2:], columns=['petal_length', 'petal_width'])

Cointegration = False
if Cointegration:
    input_dir = '../files/momentum_adj'
    output_dir = '../files/Clustering_adj/Cointegration'
    # input_dir = '../files/momentum_adj_close'
    # output_dir = '../files/Clustering_adj_close/Cointegration'

    files = sorted(filename for filename in os.listdir(input_dir))
    for file in files:
        print(file)
        # year = int(file[:4])
        # if year < 2020:
        #     continue

        data = read_and_preprocess_data(input_dir, file)


        def read_mom_data(data):
            # mom1 save and data Normalization
            mom1 = data.values.astype(float)[:, 0]
            data_normalized = (data - data.mean()) / data.std()
            mat = data_normalized.values.astype(float)

            # mom1을 제외한 mat/PCA(2-49)
            # mat = np.delete(mat, 0, axis=1)

            # mom49를 제외한 mat/PCA(1-48)
            mat = np.delete(mat, 48, axis=1)

            df_combined = pd.DataFrame(mat)
            df_combined.insert(0, 'Mom1', mom1)
            df_combined.index = data.index

            return df_combined.T


        mom_data = read_mom_data(data)




        def find_cointegrated_pairs2(data: pd.DataFrame):
            data = data.iloc[1:, :]

            start_time=time.time()

            pairs = list(combinations(data.columns, 2))  # 모든 회사 조합

            def cointegrate(pair):
                x=data[pair[0]].values
                y=data[pair[1]].values
                _, p_value, _ = coint(x, y)
                return pair, p_value
            print('1')
            import os
            max_threads=os.cpu_count()
            print(max_threads)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                results=list(executor.map(cointegrate, pairs))
            print('2')
            pair_p_values = sorted(results, key=lambda x: x[1])

            print(len(pairs))
            print(len(pair_p_values))

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution Time: {execution_time:.2f} seconds")

            start_time = time.time()
            invest_list = []
            for i, pair in enumerate(pairs):

                # Cointegration 검정
                score, pvalue, _ = coint(data[pair[0]], data[pair[1]])

                spread = data[pair[0]] - data[pair[1]]
                adf_result = sm.tsa.adfuller(spread)

                if adf_result[1] > 0.05:
                    continue

                if pvalue > 0.01:
                    continue

                mean_spread = spread.mean()
                std_spread = spread.std()
                z_score = (spread - mean_spread) / std_spread

                if float(z_score[0]) < -2:
                    pair2 = [pair[1], pair[0]]

                elif float(z_score[0]) > 2:
                    pair2 = pair

                else:
                    continue

                if pair in pairs:
                    print('spread is stationary.')
                    print('spread can be conintegrated.')
                    print('absolute z_score over 1.5')
                    print(f'{pair} is first.')
                    invest_list.append(pair2)
                    pairs = [p for p in pairs if all(item not in pair for item in p)]
                    print(len(pairs))
                else:
                    continue

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution Time: {execution_time:.2f} seconds")

            start_time = time.time()
            pairs = sorted(set([item for sublist in pairs for item in sublist]))

            LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

            for cluster_num, firms in enumerate(invest_list):
                # Sort firms based on momentum_1
                long_short = [0] * 2
                long_short[0] = -1
                long_short[1] = 1
                # Add the data to the new table
                for i, firm in enumerate(firms):
                    LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], long_short[i], cluster_num]

            for i, firm in enumerate(pairs):
                LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], 0, -1]

            firm_list_after = list(LS_table['Firm Name'])
            firm_list_before = list(mom_data.T.index)
            Missing = [item for item in firm_list_before if item not in firm_list_after]

            for i, firm in enumerate(Missing):
                LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], 0, -1]

            LS_table.sort_values(by='Cluster Index', inplace=True)

            # Save the output to a CSV file in the output directory
            LS_table.to_csv(os.path.join(output_dir, file), index=False)
            print(output_dir)

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution Time: {execution_time:.2f} seconds")


        start_time = time.time()

        find_cointegrated_pairs2(mom_data)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.2f} seconds")

        ''' def find_cointegrated_pairs(data: pd.DataFrame):
            data = data.iloc[:, 1:]
            n = data.shape[1]
            score_matrix = np.zeros((n, n))
            pvalue_matrix = np.ones((n, n))
            keys = data.keys()
            pairs = []

            # result
            stock1 = []
            stock2 = []
            pvalue_list = []
            check_95 = []
            check_98 = []

            for i in range(n):
                for j in range(i + 1, n):
                    S1 = data[keys[i]]
                    S2 = data[keys[j]]
                    result = coint(S1, S2)
                    score = result[0]
                    pvalue = result[1]
                    score_matrix[i, j] = score
                    pvalue_matrix[i, j] = pvalue

                    if pvalue < 0.05:
                        pairs.append((keys[i], keys[j]))
                        check_95.append('Y')
                    else:
                        check_95.append('N')

                    if pvalue < 0.02:
                        check_98.append('Y')
                    else:
                        check_98.append('N')

                    # result
                    stock1.append(keys[i])
                    stock2.append(keys[j])
                    pvalue_list.append(pvalue)

            pair_pvalue = pd.DataFrame()
            pair_pvalue['s1'] = stock1
            pair_pvalue['s2'] = stock2
            pair_pvalue['pvalue'] = pvalue_list
            pair_pvalue['check_95'] = check_95
            pair_pvalue['check_98'] = check_98

            pair_pvalue.sort_values('pvalue', ascending=True, inplace=True)  # ascending=True 오름차순

            return score_matrix, pvalue_matrix, pair_pvalue, pairs

        start_time = time.time()

        scores, pvalues, pair_pvalue, pairs = find_cointegrated_pairs(mom_data)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.2f} seconds")


        def save_cointegration(mom_data, pairs, pair_pvalue: pd.DataFrame, output_dir, file):
            invest_list = []

            for i in range(len(pair_pvalue.index)):
                pair = list(pair_pvalue.iloc[i, 0:2])

                spread = mom_data[pair[0]] - mom_data[pair[1]]
                adf_result = sm.tsa.adfuller(spread)

                if adf_result[1] > 0.05:
                    continue

                if pair_pvalue.iloc[i, 2] > 0.02:
                    continue

                mean_spread = spread.mean()
                std_spread = spread.std()
                z_score = (spread - mean_spread) / std_spread

                if float(z_score.iloc[0]) < -1.5:
                    pair2 = [pair[1], pair[0]]

                elif float(z_score.iloc[0]) > 1.5:
                    pair2 = pair

                else:
                    continue

                if tuple(pair) in pairs:
                    print('spread is stationary.')
                    print('spread can be conintegrated.')
                    print('absolute z_score over 1.5')
                    print(f'{pair} is first.')
                    invest_list.append(pair2)
                    pairs = [p for p in pairs if all(item not in pair for item in p)]
                    print(pairs)
                else:
                    continue

            pairs = sorted(set([item for sublist in pairs for item in sublist]))

            LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

            for cluster_num, firms in enumerate(invest_list):
                # Sort firms based on momentum_1
                long_short = [0] * 2
                long_short[0] = -1
                long_short[1] = 1
                # Add the data to the new table
                for i, firm in enumerate(firms):
                    LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], long_short[i], cluster_num]

            for i, firm in enumerate(pairs):
                LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], 0, -1]

            firm_list_after = list(LS_table['Firm Name'])
            firm_list_before = list(mom_data.T.index)
            Missing = [item for item in firm_list_before if item not in firm_list_after]

            for i, firm in enumerate(Missing):
                LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], 0, -1]

            LS_table.sort_values(by='Cluster Index', inplace=True)

            # Save the output to a CSV file in the output directory
            LS_table.to_csv(os.path.join(output_dir, file), index=False)
            print(output_dir)

        start_time=time.time()

        save_cointegration(mom_data, pairs, pair_pvalue, output_dir, file)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.2f} seconds")'''

example = False
if example:
    # Plot K_mean cluster about individual csv file
    example1 = True
    if example1:
        # Call initial method
        Do_Clustering = C.Clustering(iris_pd)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([3])

        iris_pd['species'] = iris.target
        x_kc = Do_Clustering.test.cluster_centers_[:, 0]
        y_kc = Do_Clustering.test.cluster_centers_[:, 1]

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='petal_length', y='petal_width', hue='species', style='species', s=100, data=iris_pd)
        plt.scatter(x_kc, y_kc, s=100, color='r')
        plt.xlabel('petal_length')
        plt.ylabel('petal_width')
        plt.show()

        t_SNE('K_mean', Do_Clustering.PCA_Data, Do_Clustering.K_Mean_labels)

    example2 = True
    if example2:
        # Call initial method
        Do_Clustering = C.Clustering(iris_pd)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(0.9)

        n_clusters_ = len(set(Do_Clustering.DBSCAN_labels)) - (1 if -1 in Do_Clustering.DBSCAN_labels else 0)
        n_noise_ = list(Do_Clustering.DBSCAN_labels).count(-1)

        unique_labels = set(Do_Clustering.DBSCAN_labels)
        core_samples_mask = np.zeros_like(Do_Clustering.DBSCAN_labels, dtype=bool)
        core_samples_mask[Do_Clustering.test.core_sample_indices_] = True

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = Do_Clustering.DBSCAN_labels == k

            xy = Do_Clustering.PCA_Data[class_member_mask & core_samples_mask]
            plt.plot(
                xy.iloc[:, 0],
                xy.iloc[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=14,
            )

            xy = Do_Clustering.PCA_Data[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy.iloc[:, 0],
                xy.iloc[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,
            )

        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.show()

        t_SNE('DBSCAN', Do_Clustering.PCA_Data, Do_Clustering.DBSCAN_labels)

    example3 = True
    if example3:
        # Call initial method
        Do_Clustering = C.Clustering(iris_pd)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.Agglomerative = Do_Clustering.perform_HG(0.5)

        t_SNE('Hirarchical Agglormerative', Do_Clustering.PCA_Data, Do_Clustering.Agglomerative_labels)

    example4 = True
    if example4:
        # Call initial method
        Do_Clustering = C.Clustering(iris_pd)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.Gaussian = Do_Clustering.perform_GMM(0.1)

        t_SNE('GMM', Do_Clustering.PCA_Data, Do_Clustering.Gaussian_labels)

    example5 = True
    if example5:
        # Call initial method
        Do_Clustering = C.Clustering(iris_pd)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(0.2)

        t_SNE('OPTICS', Do_Clustering.PCA_Data, Do_Clustering.OPTIC_labels)

    example6 = True
    if example6:
        # Call initial method
        Do_Clustering = C.Clustering(iris_pd)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.menshift = Do_Clustering.perform_meanshift(0.3)

        t_SNE('meanshift', Do_Clustering.PCA_Data, Do_Clustering.menshift_labels)

    example7 = True
    if example7:
        # Call initial method
        Do_Clustering = C.Clustering(iris_pd)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN()

        t_SNE('meanshift', Do_Clustering.PCA_Data, Do_Clustering.HDBSCAN_labels)

Plot = False
if Plot:
    # file to check
    # file = '1990-01.csv'
    file = '1990-01.csv'

    # Plot K_mean cluster about individual csv file
    K_mean_Plot = True
    if K_mean_Plot:
        input_dir = '../files/momentum_adj'

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Plot = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([10])

        # Plot clustering result
        Do_Result_Plot.Plot_clusters_Kmean(Do_Clustering.K_Mean)

        # Plot t_SNE result
        for i, cluster in enumerate(Do_Clustering.K_Mean):
            t_SNE('K-mean', df_combined, Do_Clustering.K_Mean_labels)

            # for i, cluster in enumerate(Do_Clustering.K_Mean):
            Do_Result_Plot.LS_Table_Save(cluster, '../files/Clustering_adj/K_Means_outlier', file)
        # Do_Result_Plot.Reversal_Table_Save(data, '../files/Clustering_adj/Reversal', file)
    # hyper parameter K(3,5,10,50,100,500,1000,1500) should be tested manually.(paper follow)

    # Plot DBSCAN cluster about individual csv file
    dbscan_Plot = True
    if dbscan_Plot:
        input_dir = '../files/momentum_adj'

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Plot = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(0.8)

        # Plot clustering result
        Do_Result_Plot.Plot_clusters(Do_Clustering.DBSCAN)

        # Plot t_SNE result
        t_SNE('DBSCAN', df_combined, Do_Clustering.DBSCAN_labels)

        # Do_Result_Plot.LS_Table_Save(Do_Clustering.DBSCAN, '../files/Clustering_adj/DBSCAN', file)
    # hyper parameter eps percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)

    # Plot HDBSCAN cluster about individual csv file
    hdbscan_Plot = True
    if hdbscan_Plot:
        input_dir = '../files/momentum_adj'

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Plot = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN()

        # Plot clustering result
        Do_Result_Plot.Plot_clusters(Do_Clustering.HDBSCAN)

        # Plot t_SNE result
        t_SNE('HDBSCAN', df_combined, Do_Clustering.HDBSCAN_labels)

        # Do_Result_Plot.LS_Table_Save(Do_Clustering.DBSCAN, '../files/Clustering_adj/DBSCAN', file)
    # hyper parameter eps percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)

    # Plot Agglomerative cluster about individual csv file
    Agglormerative_Plot = True
    if Agglormerative_Plot:
        input_dir = '../files/momentum_adj'

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Plot = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        # Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([4])
        Do_Clustering.Agglomerative = Do_Clustering.perform_HG(0.4)

        # Plot clustering result
        Do_Result_Plot.Plot_clusters(Do_Clustering.Agglomerative)

        # Plot t_SNE result
        t_SNE('Hirarchical Agglomerative', df_combined, Do_Clustering.Agglomerative_labels)

        # compare cluster result
        # analysis_clustering_result(Do_Clustering.PCA_Data, Do_Clustering.Agglomerative_labels, Do_Clustering.K_Mean_labels)

        # Do_Result_Plot.LS_Table_Save(Do_Clustering.Agglomerative, '../files/Clustering_adj/Hierarchical_Agglomerative',file)
    # hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)

    # Plot BGM cluster about individual csv file
    BGM_Plot = True
    if BGM_Plot:
        input_dir = '../files/momentum_adj'

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Plot = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.Gaussian = Do_Clustering.perform_GMM(0.15)

        # Plot clustering result
        Do_Result_Plot.Plot_clusters(Do_Clustering.Gaussian)

        # Plot t_SNE result
        t_SNE('GMM', df_combined, Do_Clustering.Gaussian_labels)

        # Do_Result_Plot.LS_Table_Save(Do_Clustering.Gaussian, '../files/Clustering_adj/Gaussian_Mixture_Model',file)
    # hyper parameter outlier probability range(0.05, 0.15, 0.01) should be tested manually.

    # Plot OPTICS cluster about individual csv file
    optics_Plot = True
    if optics_Plot:
        input_dir = '../files/momentum_adj'

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Plot = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(0.5)

        # Plot clustering result
        Do_Result_Plot.Plot_clusters(Do_Clustering.OPTIC)

        # Plot t_SNE result
        t_SNE('OPTICS', df_combined, Do_Clustering.OPTIC_labels)

        # Do_Result_Plot.LS_Table_Save(Do_Clustering.OPTIC, '../files/Clustering_adj/OPTICS', file)
    # hyper parameter percentile of min_sample[0.01, 0.05, range(0.1, 0.9, 0.1)] should be tested manually.

    # Plot Mean Shift cluster about individual csv file
    meanshift_Plot = True
    if meanshift_Plot:
        input_dir = '../files/momentum_adj'

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Plot = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.menshift = Do_Clustering.perform_meanshift(0.3)

        # # Plot clustering result
        # Do_Result_Plot.Plot_clusters(Do_Clustering.lab)
        #
        # # Plot t_SNE result
        # t_SNE('meanshift', df_combined, Do_Clustering.menshift_labels)

        # Do_Result_Plot.LS_Table_Save(Do_Clustering.menshift, '../files/Clustering_adj/Meanshift', file)
    # hyper parameter quantile (0.1, 0.2, 0.3, 0.4) should be tested manually.(paper follow)

    # Save K_mean clutering method LS_Tables

Save = True
if Save:
    K_mean_Save = False
    if K_mean_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/K_Means_outlier'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/K_Means_outlier'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([3])

            sum += Do_Result_Save.count_outlier(Do_Clustering.K_Mean[0])
            # print(Do_Result_Save.count_outlier(Do_Clustering.K_Mean[0]))

            # Save LS_Table CSV File
            for i, cluster in enumerate(Do_Clustering.K_Mean):
                Do_Result_Save.LS_Table_Save(cluster, output_dir, file)

        print(f'total outliers: {sum}')

    # Save DBSCAN clutering method LS_Tables
    dbscan_Save = False
    if dbscan_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/DBSCAN'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/DBSCAN'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)
            print(df_combined)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(0.8)

            sum += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)

            # Save LS_Table CSV File
            # Do_Result_Save.LS_Table_Save(Do_Clustering.DBSCAN, output_dir, file)

        print(f'total outliers: {sum}')

    # Save DBSCAN clutering method LS_Tables
    hdbscan_Save = False
    if hdbscan_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/HDBSCAN'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/HDBSCAN'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN()

            sum += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.HDBSCAN, output_dir, file)

        print(f'total outliers: {sum}')

    # Save Hirarchical Agglomerative clutering method LS_Tables
    Agglormerative_Save = False
    if Agglormerative_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/Hierarchical_Agglomerative'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/Hierarchical_Agglomerative'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.Agglomerative = Do_Clustering.perform_HG(0.4)

            sum += Do_Result_Save.count_outlier(Do_Clustering.Agglomerative)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.Agglomerative, output_dir, file)

        print(f'total outliers: {sum}')

    # Save BayesianGaussianMixture clutering method LS_Tables
    BGM_Save = False
    if BGM_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/Gaussian_Mixture_Model'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/Gaussian_Mixture_Model'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.Gaussian = Do_Clustering.perform_GMM(5)

            sum += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.Gaussian, output_dir, file)

        print(f'total outliers: {sum}')

    # Save OPTICS clutering method LS_Tables
    optics_Save = False
    if optics_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/OPTICS'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/OPTICS'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(0.05)

            sum += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.OPTIC, output_dir, file)

        print(f'total outliers: {sum}')

    # Save Mean Shift clutering method LS_Tables
    meanshift_Save = False
    if meanshift_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/Meanshift'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/Meanshift'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.menshift = Do_Clustering.perform_meanshift(0.1)

            sum += Do_Result_Save.count_outlier(Do_Clustering.menshift)
            print(sum)
            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.menshift, output_dir, file)

        print(f'total outliers: {sum}')

    # Save Reversal method LS_Tables
    Reversal_Save = False
    if Reversal_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/Reversal'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/Reversal'
        files = sorted(filename for filename in os.listdir(input_dir))

        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            Do_Result_Save = C.Result_Check_and_Save(data)

            # Save LS_Table CSV File
            Do_Result_Save.Reversal_Table_Save(data, output_dir, file)

    # Save BIRCH clutering method LS_Tables
    birch_Save = False
    if birch_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/HDBSCAN'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/BIRCH'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.BIRCH = Do_Clustering.perform_BIRCH()

            sum += Do_Result_Save.count_outlier(Do_Clustering.BIRCH)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.BIRCH, output_dir, file)

        print(f'total outliers: {sum}')

    # Save Affinity Propagation clutering method LS_Tables
    affinity_Save = True
    if affinity_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/HDBSCAN'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/Affinity_Propagation'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.BIRCH = Do_Clustering.perform_Affinity(0.5)

            sum += Do_Result_Save.count_outlier(Do_Clustering.Affinity)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.Affinity, output_dir, file)

        print(f'total outliers: {sum}')

# ToDo: Affinity Propagation
