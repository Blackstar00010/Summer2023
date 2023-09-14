import Clustering as C
from PCA_and_ETC import *

Reversal_Save = False
if Reversal_Save:
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

K_mean_Save = True
if K_mean_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/K_mean'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_kmeans(25)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.K_Mean)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.K_Mean, output_dir, file, save=True, raw=False)

dbscan_Save = False
if dbscan_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/DBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_DBSCAN(0.9)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.DBSCAN, output_dir, file, save=True, raw=False)

Agglormerative_Save = False
if Agglormerative_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Agglomerative'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_HA(0.8)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Agglomerative)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.Agglomerative, output_dir, file, save=True, raw=False)

hdbscan_Save = False
if hdbscan_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/HDBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_HDBSCAN(0.6)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.HDBSCAN, output_dir, file, save=True, raw=False)

optics_Save = False
if optics_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/OPTICS'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_OPTICS(0.9)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.OPTIC, output_dir, file, save=True, raw=False)

birch_Save = True
if birch_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/BIRCH'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_BIRCH(0.9)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.BIRCH)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.BIRCH, output_dir, file, save=True, raw=False)

meanshift_Save = False
if meanshift_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Meanshift'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_meanshift(0.8)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.meanshift)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.meanshift, output_dir, file, save=True, raw=False)

GMM_Save = False
if GMM_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/GMM'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_GMM(50)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.Gaussian, output_dir, file, save=True, raw=False)
