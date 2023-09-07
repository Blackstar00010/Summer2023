import Clustering as C
from PCA_and_ETC import *

# Save Reversal method LS_Tables
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

# Save K_mean clutering method LS_Tables
# hyper parameter K(1,2,3,4,5,10,50,100,500,1000) should be tested manually.(paper follow)
K_mean_Save = False
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
        Do_Clustering.perform_kmeans(200)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.K_Mean)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.K_Mean, output_dir, file, save=True, raw=False)

# Save DBSCAN clutering method LS_Tables
# hyper parameter eps percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
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
        Do_Clustering.perform_DBSCAN(0.8)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.DBSCAN, output_dir, file, save=True, raw=False)

# Save Hirarchical Agglomerative clutering method LS_Tables
# hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
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
        Do_Clustering.perform_HA(0.9)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Agglomerative)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.Agglomerative, output_dir, file, save=True, raw=False)

# Save HDBSCAN clutering method LS_Tables
# hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.
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
        Do_Clustering.perform_HDBSCAN(0.9)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.HDBSCAN, output_dir, file, save=True, raw=False)

# Save OPTICS clutering method LS_Tables
# hyper parameter percentile of xi range(0.05, 0.09, 0.01) should be tested manually.
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

# Save BIRCH clutering method LS_Tables
# hyper parameter percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
birch_Save = False
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

# Save Meanshift clutering method LS_Tables
# hyper parameter quantile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)
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
        Do_Clustering.perform_meanshift(0.9)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.meanshift)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.meanshift, output_dir, file, save=True, raw=False)

# Save GaussianMixture clutering method LS_Tables
# hyper parameter outlier probability [1, 5, 10, 15, 20] should be tested manually.
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
        Do_Clustering.perform_GMM(30)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.Gaussian, output_dir, file, save=True, raw=False)
