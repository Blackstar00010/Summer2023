import Clustering as C
from PCA_and_ETC import *
from sklearn.metrics import silhouette_score

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
    input_dir = '../files/momentum_adj_close'
    output_dir = '../files/Clustering_adj_close/K_Means_outlier'
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
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.K_Mean = Do_Clustering.perform_kmeans(2)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.K_Mean[0])

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
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save DBSCAN clutering method LS_Tables
# hyper parameter eps percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
dbscan_Save = True
if dbscan_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/DBSCAN'

    input_dir = '../files/momentum_adj_close'
    output_dir = '../files/Clustering_adj_close/DBSCAN'
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
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(0.6)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.DBSCAN, output_dir, file)

        silhouette_avg = silhouette_score(df_combined, Do_Clustering.DBSCAN_labels)
        sil += silhouette_avg
        cl += len(sorted(list(set(Do_Clustering.DBSCAN_labels))))
        print("The average silhouette score is:", silhouette_avg)
        print("Number of clusters is:", len(sorted(list(set(Do_Clustering.DBSCAN_labels)))))

    sil = sil / 312
    cl = cl / 312
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save HDBSCAN clutering method LS_Tables
# hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.
hdbscan_Save = True
if hdbscan_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/HDBSCAN'

    input_dir = '../files/momentum_adj_close'
    output_dir = '../files/Clustering_adj_close/HDBSCAN'
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
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN(0.5)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

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
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save Hirarchical Agglomerative clutering method LS_Tables
# hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
Agglormerative_Save = True
if Agglormerative_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/Hierarchical_Agglomerative'

    input_dir = '../files/momentum_adj_close'
    output_dir = '../files/Clustering_adj_close/Hierarchical_Agglomerative'
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
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.Agglomerative = Do_Clustering.perform_HA(0.5)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

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
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save GaussianMixture clutering method LS_Tables
# hyper parameter outlier probability [1, 5, 10, 15, 20] should be tested manually.
GMM_Save = True
if GMM_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/Gaussian_Mixture_Model'

    input_dir = '../files/momentum_adj_close'
    output_dir = '../files/Clustering_adj_close/Gaussian_Mixture_Model'
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
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.Gaussian = Do_Clustering.perform_GMM(1)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)

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
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save OPTICS clutering method LS_Tables
# hyper parameter percentile of xi range(0.05, 0.09, 0.01) should be tested manually.
optics_Save = True
if optics_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/OPTICS'

    input_dir = '../files/momentum_adj_close'
    output_dir = '../files/Clustering_adj_close/OPTICS'
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
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(0.7)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)

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
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save Meanshift clutering method LS_Tables
# hyper parameter quantile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
meanshift_Save = True
if meanshift_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/Meanshift'

    input_dir = '../files/momentum_adj_close'
    output_dir = '../files/Clustering_adj_close/Meanshift'
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
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.menshift = Do_Clustering.perform_meanshift(0.9)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.menshift)

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
    print('average number of clusters:', cl)
    print('silhouette score:', sil)
    print(f'total outliers: {outliers_count}')

# Save BIRCH clutering method LS_Tables
# hyper parameter percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
birch_Save = True
if birch_Save:
        # input_dir = '../files/momentum_adj'
        # output_dir ='../files/Clustering_adj/HDBSCAN'

        input_dir = '../files/momentum_adj_close'
        output_dir = '../files/Clustering_adj_close/BIRCH'
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
            Do_Result_Save = C.Result_Check_and_Save(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.BIRCH = Do_Clustering.perform_BIRCH(0.7)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.BIRCH)

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
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {outliers_count}')
