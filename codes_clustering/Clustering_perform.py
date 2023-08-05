import warnings
import Clustering as C
from PCA_and_ETC import *

# file to check
file = '1990-01.csv'

# turn off warning
warnings.filterwarnings("ignore")

# Plot K_mean cluster about individual csv file
K_mean_Plot = False
if K_mean_Plot:
    input_dir = '../files/momentum_adj'

    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.Result_Check_and_Save(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([4])

    # Plot clustering result
    Do_Result_Plot.Plot_clusters_Kmean(Do_Clustering.K_Mean)

    # Plot t_SNE result
    for i, cluster in enumerate(Do_Clustering.K_Mean):
        t_SNE('K-mean', df_combined, Do_Clustering.K_Mean_labels)

    # for i, cluster in enumerate(Do_Clustering.K_Mean):
    #     Do_Result_Plot.LS_Table_Save(cluster, '../files/Clustering_adj/K_Means_outlier', file)
    # Do_Result_Plot.Reversal_Table_Save(data, '../files/Clustering_adj/Reversal', file)
# hyper parameter K(3,5,10,50,100,500,1000,1500) should be tested manually.(paper follow)

# Plot DBSCAN cluster about individual csv file
dbscan_Plot = False
if dbscan_Plot:
    input_dir = '../files/momentum_adj'

    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.Result_Check_and_Save(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([4])
    Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(0.8)

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.DBSCAN)

    # Plot t_SNE result
    t_SNE('DBSCAN', df_combined, Do_Clustering.DBSCAN_labels)

    # compare cluster result
    analysis_clustering_result(Do_Clustering.PCA_Data, Do_Clustering.DBSCAN_labels, Do_Clustering.K_Mean_labels)

    # Do_Result_Plot.LS_Table_Save(Do_Clustering.DBSCAN, '../files/Clustering_adj/DBSCAN', file)
# hyper parameter eps percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)

# Plot HDBSCAN cluster about individual csv file
hdbscan_Plot = False
if hdbscan_Plot:
    input_dir = '../files/momentum_adj'

    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.Result_Check_and_Save(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN(0.2)

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.HDBSCAN)

    # Plot t_SNE result
    t_SNE('HDBSCAN', df_combined, Do_Clustering.HDBSCAN_labels)

    # Do_Result_Plot.LS_Table_Save(Do_Clustering.DBSCAN, '../files/Clustering_adj/DBSCAN', file)
# hyper parameter eps percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)

# Plot Agglomerative cluster about individual csv file
Agglormerative_Plot = False
if Agglormerative_Plot:
    input_dir = '../files/momentum_adj'

    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.Result_Check_and_Save(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([4])
    Do_Clustering.Agglomerative = Do_Clustering.perform_HG(0.6)

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.Agglomerative)

    # Plot t_SNE result
    t_SNE('Hirarchical Agglomerative', df_combined, Do_Clustering.Agglomerative_labels)

    # compare cluster result
    analysis_clustering_result(Do_Clustering.PCA_Data, Do_Clustering.Agglomerative_labels, Do_Clustering.K_Mean_labels)

    # Do_Result_Plot.LS_Table_Save(Do_Clustering.Agglomerative, '../files/Clustering_adj/Hierarchical_Agglomerative',file)
# hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)

# Plot BGM cluster about individual csv file
BGM_Plot = False
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
    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([4])

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.Gaussian)

    # Plot t_SNE result
    t_SNE('GMM', df_combined, Do_Clustering.Gaussian_labels)

    # compare cluster result
    analysis_clustering_result(Do_Clustering.PCA_Data, Do_Clustering.Gaussian_labels, Do_Clustering.K_Mean_labels)

    # Do_Result_Plot.LS_Table_Save(Do_Clustering.Gaussian, '../files/Clustering_adj/Gaussian_Mixture_Model',file)
# hyper parameter outlier probability range(0.05, 0.15, 0.01) should be tested manually.

# Plot OPTICS cluster about individual csv file
optics_Plot = False
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
meanshift_Plot = False
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

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.lab)

    # Plot t_SNE result
    t_SNE('meanshift', df_combined, Do_Clustering.menshift_labels)

    # Do_Result_Plot.LS_Table_Save(Do_Clustering.menshift, '../files/Clustering_adj/Meanshift', file)
# hyper parameter quantile (0.1, 0.2, 0.3, 0.4) should be tested manually.(paper follow)

# Save K_mean clutering method LS_Tables
K_mean_Save = False
if K_mean_Save:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))

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

        # Save LS_Table CSV File
        for i, cluster in enumerate(Do_Clustering.K_Mean):
            Do_Result_Save.LS_Table_Save(cluster, '../files/Clustering_adj/K_Means_outlier', file)

# Save DBSCAN clutering method LS_Tables
dbscan_Save = False
if dbscan_Save:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))

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

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.DBSCAN, '../files/Clustering_adj/DBSCAN', file)

# Save DBSCAN clutering method LS_Tables
hdbscan_Save = False
if hdbscan_Save:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN(0.8)

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.HDBSCAN, '../files/Clustering_adj/HDBSCAN', file)

# Save Hirarchical Agglomerative clutering method LS_Tables
Agglormerative_Save = False
if Agglormerative_Save:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.Agglomerative = Do_Clustering.perform_HG(0.6)

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.Agglomerative, '../files/Clustering_adj/Hierarchical_Agglomerative',file)

# Save BayesianGaussianMixture clutering method LS_Tables
BGM_Save = False
if BGM_Save:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.Gaussian = Do_Clustering.perform_GMM(0.15)

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.Gaussian, '../files/Clustering_adj/Gaussian_Mixture_Model', file)

# Save OPTICS clutering method LS_Tables
optics_Save = False
if optics_Save:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))

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

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.OPTIC, '../files/Clustering_adj/OPTICS', file)

# Save Mean Shift clutering method LS_Tables
meanshift_Save = True
if meanshift_Save:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))

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

        # Save LS_Table CSV File
        Do_Result_Save.LS_Table_Save(Do_Clustering.menshift, '../files/Clustering_adj/Meanshift', file)
