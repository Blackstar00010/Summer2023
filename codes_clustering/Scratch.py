import seaborn as sns
import Clustering as C
from PCA_and_ETC import *
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

Cointegration = False
if Cointegration:
    # input_dir = '../files/momentum_adj'
    # output_dir = '../files/Clustering_adj/Cointegration'
    input_dir = '../files/momentum_adj_close'
    output_dir = '../files/Clustering_adj_close/Cointegration'

    files = sorted(filename for filename in os.listdir(input_dir))
    is_jamesd = 'jamesd' in os.path.abspath('.')
    for file in files:
        print(file)
        if file in os.listdir(output_dir):
            continue
        data = read_and_preprocess_data(input_dir, file)

        mom_data = read_mom_data(data)

        # inv_list = find_cointegrated_pairs_deprecated(mom_data)
        inv_list = find_cointegrated_pairs(mom_data)

        LS_Table = True
        if LS_Table:
            save_cointegrated_LS(output_dir, file, mom_data, inv_list)

example = False
if example:
    # sample data
    iris = load_iris()
    iris_pd = pd.DataFrame(iris.data[:, 2:], columns=['petal_length', 'petal_width'])

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
        # analysis_clustering_result(Do_Clustering.PCA_Data, Do_Clustering..OPTIC_labels, Do_Clustering.K_Mean_labels)

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
    number_of_cluster=[]

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

    # Save K_mean clutering method LS_Tables
    # hyper parameter K(1,2,3,4,5,10,50,100,500,1000) should be tested manually.(paper follow)
    K_mean_Save = True
    if K_mean_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/K_Means_outlier'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/K_Means_outlier'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        sil = 0
        cl = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([2])

            sum += Do_Result_Save.count_outlier(Do_Clustering.K_Mean[0])

            # Save LS_Table CSV File
            for i, cluster in enumerate(Do_Clustering.K_Mean):
                Do_Result_Save.LS_Table_Save(cluster, output_dir, file)

            silhouette_avg = silhouette_score(df_combined, Do_Clustering.K_Mean_labels)
            sil += silhouette_avg
            cl += len(sorted(list(set(Do_Clustering.K_Mean_labels))))
            print("The average silhouette score is:", silhouette_avg)
            print("Number of clusters is:", len(sorted(list(set(Do_Clustering.K_Mean_labels)))))

        sil = sil / 312
        cl = cl / 312
        number_of_cluster.append(sum)
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {sum}')

    # Save DBSCAN clutering method LS_Tables
    # hyper parameter eps percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
    dbscan_Save = True
    if dbscan_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/DBSCAN'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/DBSCAN'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
        sil = 0
        cl = 0
        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(0.6)

            sum += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.DBSCAN, output_dir, file)

            silhouette_avg = silhouette_score(df_combined, Do_Clustering.DBSCAN_labels)
            sil += silhouette_avg
            cl += len(sorted(list(set(Do_Clustering.DBSCAN_labels))))
            print("The average silhouette score is:", silhouette_avg)
            print("Number of clusters is:", len(sorted(list(set(Do_Clustering.DBSCAN_labels)))))

        sil = sil / 312
        cl = cl / 312
        number_of_cluster.append(sum)
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {sum}')

    # Save HDBSCAN clutering method LS_Tables
    # hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.
    hdbscan_Save = True
    if hdbscan_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/HDBSCAN'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/HDBSCAN'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
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
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN(0.5)

            sum += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.HDBSCAN, output_dir, file)

            if len(sorted(list(set(Do_Clustering.HDBSCAN_labels)))) != 1:
                silhouette_avg = silhouette_score(df_combined, Do_Clustering.HDBSCAN_labels)
                sil += silhouette_avg
                sil_num += 1
                print("The average silhouette score is:", silhouette_avg)
            cl += len(sorted(list(set(Do_Clustering.HDBSCAN_labels))))

            print("Number of clusters is:", len(sorted(list(set(Do_Clustering.HDBSCAN_labels)))))

        sil = sil / sil_num
        cl = cl / 312
        number_of_cluster.append(sum)
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {sum}')

    # Save Hirarchical Agglomerative clutering method LS_Tables
    # hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
    Agglormerative_Save = True
    if Agglormerative_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/Hierarchical_Agglomerative'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/Hierarchical_Agglomerative'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
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
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.Agglomerative = Do_Clustering.perform_HA(0.5)

            sum += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.Agglomerative, output_dir, file)

            if len(sorted(list(set(Do_Clustering.Agglomerative_labels)))) != 1:
                silhouette_avg = silhouette_score(df_combined, Do_Clustering.Agglomerative_labels)
                sil += silhouette_avg
                sil_num += 1
                print("The average silhouette score is:", silhouette_avg)
            cl += len(sorted(list(set(Do_Clustering.Agglomerative_labels))))

            print("Number of clusters is:", len(sorted(list(set(Do_Clustering.Agglomerative_labels)))))

        sil = sil / sil_num
        cl = cl / 312
        number_of_cluster.append(sum)
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {sum}')

    # Save GaussianMixture clutering method LS_Tables
    # hyper parameter outlier probability [1, 5, 10, 15, 20] should be tested manually.
    GMM_Save = True
    if GMM_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/Gaussian_Mixture_Model'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/Gaussian_Mixture_Model'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
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
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.Gaussian = Do_Clustering.perform_GMM(1)

            sum += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.Gaussian, output_dir, file)

            if len(sorted(list(set(Do_Clustering.Gaussian_labels)))) != 1:
                silhouette_avg = silhouette_score(df_combined, Do_Clustering.Gaussian_labels)
                sil += silhouette_avg
                sil_num += 1
                print("The average silhouette score is:", silhouette_avg)
            cl += len(sorted(list(set(Do_Clustering.Gaussian_labels))))

            print("Number of clusters is:", len(sorted(list(set(Do_Clustering.Gaussian_labels)))))
        sil = sil / sil_num
        cl = cl / 312
        number_of_cluster.append(sum)
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {sum}')

    # Save OPTICS clutering method LS_Tables
    # hyper parameter percentile of xi range(0.05, 0.09, 0.01) should be tested manually.
    optics_Save = True
    if optics_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/OPTICS'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/OPTICS'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
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
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(0.7)

            sum += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.OPTIC, output_dir, file)

            if len(sorted(list(set(Do_Clustering.OPTIC_labels)))) != 1:
                silhouette_avg = silhouette_score(df_combined, Do_Clustering.OPTIC_labels)
                sil += silhouette_avg
                sil_num += 1
                print("The average silhouette score is:", silhouette_avg)

            cl += len(sorted(list(set(Do_Clustering.OPTIC_labels))))

            print("Number of clusters is:", len(sorted(list(set(Do_Clustering.OPTIC_labels)))))

        sil = sil / sil_num
        cl = cl / 312
        number_of_cluster.append(sum)
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {sum}')

    # Save Meanshift clutering method LS_Tables
    # hyper parameter quantile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
    meanshift_Save = True
    if meanshift_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/Meanshift'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/Meanshift'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
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
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.menshift = Do_Clustering.perform_meanshift(0.9)

            sum += Do_Result_Save.count_outlier(Do_Clustering.menshift)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.meanshift, output_dir, file)

            if len(sorted(list(set(Do_Clustering.meanshift_labels)))) != 1:
                silhouette_avg = silhouette_score(df_combined, Do_Clustering.meanshift_labels)
                sil += silhouette_avg
                sil_num += 1
                print("The average silhouette score is:", silhouette_avg)

            cl += len(sorted(list(set(Do_Clustering.meanshift_labels))))

            print("Number of clusters is:", len(sorted(list(set(Do_Clustering.meanshift_labels)))))

        sil = sil / sil_num
        cl = cl / 312
        number_of_cluster.append(sum)
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {sum}')

    # Save BIRCH clutering method LS_Tables
    # hyper parameter percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
    birch_Save = True
    if birch_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/HDBSCAN'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/BIRCH'
        files = sorted(filename for filename in os.listdir(input_dir))
        sum = 0
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
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.BIRCH = Do_Clustering.perform_BIRCH(0.7)

            sum += Do_Result_Save.count_outlier(Do_Clustering.BIRCH)

            # Save LS_Table CSV File
            Do_Result_Save.LS_Table_Save(Do_Clustering.BIRCH, output_dir, file)

            if len(sorted(list(set(Do_Clustering.BIRCH_labels)))) != 1:
                silhouette_avg = silhouette_score(df_combined, Do_Clustering.BIRCH_labels)
                sil += silhouette_avg
                sil_num += 1
                print("The average silhouette score is:", silhouette_avg)

            cl += len(sorted(list(set(Do_Clustering.BIRCH_labels))))

            print("Number of clusters is:", len(sorted(list(set(Do_Clustering.BIRCH_labels)))))

        sil = sil / sil_num
        cl = cl / 312
        number_of_cluster.append(sum)
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {sum}')
