import Clustering as C
from PCA_and_tSNE import *

total = False
if total:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        Do_Clustering = C.Clustering(df_combined)
        Do_Table_Generate = C.Result_Check_and_Save(df_combined)

        Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([10])
        Do_Clustering.Gaussian = Do_Clustering.GMM(0.1)
        Do_Clustering.Agglomerative = Do_Clustering.HG(0.5)
        Do_Clustering.OPTIC = Do_Clustering.OPTICS()
        Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN()

        for i, cluster in enumerate(Do_Clustering.K_Mean):
            Do_Table_Generate.new_table_generate(cluster, '../files/Clustering_adj/K_Means_outlier', file)
        Do_Table_Generate.new_table_generate(Do_Clustering.DBSCAN, '../files/Clustering_adj/DBSCAN', file)
        Do_Table_Generate.new_table_generate(Do_Clustering.Gaussian, '../files/Clustering_adj/Gaussian_Mixture_Model',
                                             file)
        Do_Table_Generate.new_table_generate(Do_Clustering.Agglomerative,
                                             '../files/Clustering_adj/Hierarchical_Agglomerative', file)
        Do_Table_Generate.new_table_generate(Do_Clustering.OPTIC, '../files/Clustering_adj/OPTICS', file)
        Do_Table_Generate.reversal_table_generate(data, '../files/Clustering_adj/Reversal', file)

K_Mean = False
if K_Mean:
    input_dir = '../files/momentum_adj'
    file = '2018-01.csv'
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    Do_Clustering = C.Clustering(df_combined)
    Do_Table_Generate = C.Result_Check_and_Save(df_combined)

    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([10])

    Do_Table_Generate.plot_clusters_KMean(Do_Clustering.K_Mean)
    for i, cluster in enumerate(Do_Clustering.K_Mean):
        t_SNE(df_combined, Do_Clustering.K_Mean_labels)

Hierachical = False
if Hierachical:
    input_dir = '../files/momentum_adj'
    file = '2018-01.csv'
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    Do_Clustering = C.Clustering(df_combined)
    Do_Table_Generate = C.Result_Check_and_Save(df_combined)

    Do_Clustering.Agglomerative = Do_Clustering.HG(0.5)

    Do_Table_Generate.plot_clusters(Do_Clustering.Agglomerative)
    t_SNE(df_combined, Do_Clustering.Agglomerative_labels)

dbscan = False
if dbscan:
    input_dir = '../files/momentum_adj'
    file = '2018-01.csv'
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    Do_Clustering = C.Clustering(df_combined)
    Do_Table_Generate = C.Result_Check_and_Save(df_combined)

    Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN()

    Do_Table_Generate.plot_clusters(Do_Clustering.DBSCAN)
    t_SNE(df_combined, Do_Clustering.DBSCAN_labels)

GMM = False
if GMM:
    input_dir = '../files/momentum_adj'
    file = '2018-01.csv'
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    Do_Clustering = C.Clustering(df_combined)
    Do_Table_Generate = C.Result_Check_and_Save(df_combined)

    Do_Clustering.Gaussian = Do_Clustering.GMM(0.1)

    Do_Table_Generate.plot_clusters(Do_Clustering.Gaussian)
    t_SNE(df_combined, Do_Clustering.Gaussian_labels)

optics = True
if optics:
    input_dir = '../files/momentum_adj'
    file = '2018-01.csv'
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    Do_Clustering = C.Clustering(df_combined)
    Do_Table_Generate = C.Result_Check_and_Save(df_combined)

    Do_Clustering.OPTIC = Do_Clustering.OPTICS()

    Do_Table_Generate.plot_clusters(Do_Clustering.OPTIC)
    t_SNE(df_combined, Do_Clustering.OPTIC_labels)
