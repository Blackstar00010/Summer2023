from sklearn import metrics

import Clustering as C
from PCA_and_tSNE import *

# file to check
file = '1993-04.csv'

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
    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([10])

    # Plot clustering result
    Do_Result_Plot.Plot_clusters_Kmean(Do_Clustering.K_Mean)

    # Plot t_SNE result
    for i, cluster in enumerate(Do_Clustering.K_Mean):
        t_SNE(df_combined, Do_Clustering.K_Mean_labels)

# Plot DBSCAN cluster about individual csv file
dbscan = True
if dbscan:
    input_dir = '../files/momentum_adj'

    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.Result_Check_and_Save(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([10])
    Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN()

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.perform_DBSCAN())

    # Plot t_SNE result
    t_SNE(df_combined, Do_Clustering.DBSCAN_labels)

    print(f"Homogeneity: {metrics.homogeneity_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}")
    print(f"Completeness: {metrics.completeness_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}")
    print(f"V-measure: {metrics.v_measure_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}")
    print(
        f"Adjusted Rand Index: {metrics.adjusted_rand_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}")
    print(
        "Adjusted Mutual Information:"
        f" {metrics.adjusted_mutual_info_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}"
    )
    print(
        f"Silhouette Coefficient: {metrics.silhouette_score(Do_Clustering.PCA_Data, Do_Clustering.DBSCAN_labels):.3f}")

# Plot DBSCAN cluster about individual csv file
dbscan2 = False
if dbscan2:
    input_dir = '../files/momentum_adj'

    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.Result_Check_and_Save(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([10])
    Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN2()

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.perform_DBSCAN2())

    # Plot t_SNE result
    t_SNE(df_combined, Do_Clustering.DBSCAN_labels)

    print(f"Homogeneity: {metrics.homogeneity_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}")
    print(f"Completeness: {metrics.completeness_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}")
    print(f"V-measure: {metrics.v_measure_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}")
    print(
        f"Adjusted Rand Index: {metrics.adjusted_rand_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}")
    print(
        "Adjusted Mutual Information:"
        f" {metrics.adjusted_mutual_info_score(Do_Clustering.K_Mean_labels, Do_Clustering.DBSCAN_labels):.3f}"
    )
    print(
        f"Silhouette Coefficient: {metrics.silhouette_score(Do_Clustering.PCA_Data, Do_Clustering.DBSCAN_labels):.3f}")

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
    Do_Clustering.Agglomerative = Do_Clustering.perform_HG(0.6)

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.Agglomerative)

    # Plot t_SNE result
    t_SNE(df_combined, Do_Clustering.Agglomerative_labels)

# Plot GMM cluster about individual csv file
GMM = True
if GMM:
    input_dir = '../files/momentum_adj'

    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.Result_Check_and_Save(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.Gaussian = Do_Clustering.perform_GMM(0.1)

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.Gaussian)

    # Plot t_SNE result
    t_SNE(df_combined, Do_Clustering.Gaussian_labels)

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
    Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS()

    # Plot clustering result
    Do_Result_Plot.Plot_clusters(Do_Clustering.OPTIC)

    # Plot t_SNE result
    t_SNE(df_combined, Do_Clustering.OPTIC_labels)

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
