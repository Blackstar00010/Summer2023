import Clustering as C
from PCA_and_ETC import *
from sklearn.metrics import silhouette_score

# Save K_mean clutering method LS_Tables
# hyper parameter K(1,2,3,4,5,10,50,100,500,1000) should be tested manually.(paper follow)
K_mean_Save = False
if K_mean_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/K_Means_outlier'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0
    for i in [2,3,4,5,10,50,100,500,1000]:
        print(i)

        for file in files:
            print(file)

            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            df_combined = generate_PCA_Data(data)

            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.K_Mean = Do_Clustering.perform_kmeans(i,0.5)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.K_Mean[0])

            cl += len(sorted(list(set(Do_Clustering.K_Mean_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.K_Mean_labels)))

        cl = cl / len(files)
        print('average number of clusters:', cl)
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

        cl += len(sorted(list(set(Do_Clustering.DBSCAN_labels))))
        print("Number of clusters is:", len(set(Do_Clustering.DBSCAN_labels)))

    cl = cl / len(files)
    print('average number of clusters:', cl)
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

        cl += len(sorted(list(set(Do_Clustering.HDBSCAN_labels))))
        print("Number of clusters is:", len(set(Do_Clustering.HDBSCAN_labels)))

    cl = cl / len(files)
    print('average number of clusters:', cl)
    print(f'total outliers: {outliers_count}')

# Save Hirarchical Agglomerative clutering method LS_Tables
# hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
Agglormerative_Save = True
if Agglormerative_Save:
    # input_dir = '../files/momentum_adj'
    # output_dir ='../files/Clustering_adj/Hierarchical_Agglomerative'

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Hierarchical_Agglomerative'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0

    for i in np.arange(0.1, 1, 0.1):

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
            Do_Clustering.Agglomerative = Do_Clustering.perform_HA(i, draw_dendro=False)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Agglomerative)

            t=Do_Result_Save.ls_table(Do_Clustering.Agglomerative, output_dir, file, save=False, raw=False)

            cl += len(sorted(list(set(Do_Clustering.Agglomerative_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.Agglomerative_labels)))

        cl = cl / len(files)
        print('average number of clusters:', cl)
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

        cl += len(sorted(list(set(Do_Clustering.Gaussian_labels))))
        print("Number of clusters is:", len(set(Do_Clustering.Gaussian_labels)))

    cl = cl / len(files)
    print('average number of clusters:', cl)
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

        cl += len(sorted(list(set(Do_Clustering.OPTIC_labels))))
        print("Number of clusters is:", len(set(Do_Clustering.OPTIC_labels)))

    cl = cl / len(files)
    print('average number of clusters:', cl)
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

        cl += len(sorted(list(set(Do_Clustering.meanshift_labels))))
        print("Number of clusters is:", len(set(Do_Clustering.meanshift_labels)))

    cl = cl / len(files)
    print('average number of clusters:', cl)
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

        cl += len(sorted(list(set(Do_Clustering.BIRCH_labels))))
        print("Number of clusters is:", len(set(Do_Clustering.BIRCH_labels)))

    cl = cl / len(files)
    print('average number of clusters:', cl)
    print(f'total outliers: {outliers_count}')
