import warnings
import Clustering as C
from PCA_and_ETC import *

# Save K_mean clutering method LS_Tables
K_mean_Save = False
if K_mean_Save:
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
        Do_Clustering.K_Mean = Do_Clustering.perform_kmeans(10)

        sum += Do_Result_Save.count_outlier(Do_Clustering.K_Mean)

        # Save LS_Table CSV File
        for i, cluster in enumerate(Do_Clustering.K_Mean):
            Do_Result_Save.LS_Table_Save(cluster, '../files/Clustering_adj_close/K_Means_outlier', file)

    print(f'total outliers: {sum}')

# Save DBSCAN clutering method LS_Tables
dbscan_Save = False
if dbscan_Save:
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
        Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(0.8)

        sum += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.DBSCAN, '../files/Clustering_adj_close/DBSCAN', file)

    print(f'total outliers: {sum}')

# Save DBSCAN clutering method LS_Tables
hdbscan_Save = False
if hdbscan_Save:
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
        Do_Result_Save.LS_Table_Save(Do_Clustering.HDBSCAN, '../files/Clustering_adj_close/HDBSCAN', file)

    print(f'total outliers: {sum}')

# Save Hirarchical Agglomerative clutering method LS_Tables
Agglomerative_Save = False
if Agglomerative_Save:
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
        Do_Clustering.Agglomerative = Do_Clustering.perform_HA(0.4)

        sum += Do_Result_Save.count_outlier(Do_Clustering.Agglomerative)

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.Agglomerative,
                                     '../files/Clustering_adj_close/Hierarchical_Agglomerative', file)

    print(f'total outliers: {sum}')

# Save BayesianGaussianMixture clutering method LS_Tables
BGM_Save = False
if BGM_Save:
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
        Do_Clustering.Gaussian = Do_Clustering.perform_GMM(0.1)

        sum += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.Gaussian, '../files/Clustering_adj_close/Gaussian_Mixture_Model',
                                     file)

    print(f'total outliers: {sum}')

# Save OPTICS clutering method LS_Tables
optics_Save = False
if optics_Save:
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
        Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(0.5)

        sum += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.OPTIC, '../files/Clustering_adj_close/OPTICS', file)

    print(f'total outliers: {sum}')

# Save Mean Shift clutering method LS_Tables
meanshift_Save = False
if meanshift_Save:
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
        Do_Clustering.menshift = Do_Clustering.perform_meanshift(0.3)

        sum += Do_Result_Save.count_outlier(Do_Clustering.menshift)

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.menshift, '../files/Clustering_adj_close/Meanshift', file)

    print(f'total outliers: {sum}')

# Save Reversal method LS_Tables
Reversal_Save = False
if Reversal_Save:
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        Do_Result_Save = C.Result_Check_and_Save(data)

        # Save LS_Table CSV File
        Do_Result_Save.Reversal_Table_Save(data, '../files/Clustering_adj_close/Reversal', file)
