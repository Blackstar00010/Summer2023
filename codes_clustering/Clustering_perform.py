import warnings
import Clustering as C
from PCA_and_ETC import *

# file to check
file = '1991-12.csv'
# file = '2020-02.csv'

# turn off warning
warnings.filterwarnings("ignore")

# Plot K_mean cluster about individual csv file
K_Mean = False
if K_Mean:
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
# hyper parameter K(3,5,10,50,100,500,1000,1500) should be tested manually.(paper follow)

# Plot DBSCAN cluster about individual csv file
dbscan = False
if dbscan:
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
    Do_Result_Plot.Plot_clusters(Do_Clustering.perform_DBSCAN(0.8))

    # Plot t_SNE result
    t_SNE('DBSCAN', df_combined, Do_Clustering.DBSCAN_labels)

    # compare cluster result
    analysis_clustering_result(Do_Clustering.PCA_Data, Do_Clustering.DBSCAN_labels, Do_Clustering.K_Mean_labels)
# hyper parameter eps percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)

# Plot Agglomerative cluster about individual csv file
Hierachical = False
if Hierachical:
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

    # # Plot clustering result
    # Do_Result_Plot.Plot_clusters(Do_Clustering.Agglomerative)

    # Plot t_SNE result
    t_SNE('Hirarchical Agglomerative', df_combined, Do_Clustering.Agglomerative_labels)

    # compare cluster result
    analysis_clustering_result(Do_Clustering.PCA_Data, Do_Clustering.Agglomerative_labels, Do_Clustering.K_Mean_labels)
# hyper parameter distance percentile range(0.1, 0.9, 0.1) should be tested manually.(paper follow)

# Plot GMM cluster about individual csv file
GMM = False
if GMM:
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
# hyper parameter outlier probability range(0.05, 0.15, 0.01) should be tested manually.

# Plot OPTICS cluster about individual csv file
optics = False
if optics:
    input_dir = '../files/momentum_adj'

    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.Result_Check_and_Save(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(0.9)

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.OPTIC)

    # Plot t_SNE result
    t_SNE('OPTICS', df_combined, Do_Clustering.OPTIC_labels)
# hyper parameter percentile of min_sample[0.01, 0.05, range(0.1, 0.9, 0.1)] should be tested manually.

# Plot meanshift cluster about individual csv file
meanshift = True
if meanshift:
    input_dir = '../files/momentum_adj'

    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.Result_Check_and_Save(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.menshift = Do_Clustering.perform_meanshift(0.2)

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.lab)

    # Plot t_SNE result
    t_SNE('lab', df_combined, Do_Clustering.menshift_labels)
# hyper parameter quantile (0.1, 0.2, 0.3, 0.4) should be tested manually.(paper follow)

# Save all clutering method LS_Tables
total = False
if total:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.Result_Check_and_Save(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([10])
        Do_Clustering.Gaussian = Do_Clustering.perform_GMM(0.1)
        Do_Clustering.Agglomerative = Do_Clustering.perform_HG(0.5)
        Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS()
        Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN()

        # Save LS_Table CSV File
        for i, cluster in enumerate(Do_Clustering.K_Mean):
            Do_Result_Save.LS_Table_Save(cluster, '../files/Clustering_adj/K_Means_outlier', file)
        Do_Result_Save.LS_Table_Save(Do_Clustering.DBSCAN, '../files/Clustering_adj/DBSCAN', file)
        Do_Result_Save.LS_Table_Save(Do_Clustering.Gaussian, '../files/Clustering_adj/Gaussian_Mixture_Model',
                                     file)
        Do_Result_Save.LS_Table_Save(Do_Clustering.Agglomerative,
                                     '../files/Clustering_adj/Hierarchical_Agglomerative', file)
        Do_Result_Save.LS_Table_Save(Do_Clustering.OPTIC, '../files/Clustering_adj/OPTICS', file)
        Do_Result_Save.Reversal_Table_Save(data, '../files/Clustering_adj/Reversal', file)
