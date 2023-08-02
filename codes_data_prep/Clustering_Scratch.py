import Clustering as C
from Clustering import *

input_dir = '../files/momentum_adj'
files = sorted(filename for filename in os.listdir(input_dir))
eps_values = np.linspace(0.01, 3., 100)
min_samples_values = range(2, 20)

for file in files:
    print(file)
    data=read_and_preprocess_data(input_dir, file)
    df_combined=generate_PCA_Data(data)

    Do_Clustering=C.Clustering(df_combined)
    Do_Table_Generate=C.LS_Table(df_combined)

    Do_Clustering.K_Mean=Do_Clustering.perform_kmeans([10])
    Do_Clustering.Gaussian=Do_Clustering.GMM(0.1)
    Do_Clustering.Agglomerative=Do_Clustering.HG(0.5)
    Do_Clustering.OPTIC=Do_Clustering.OPTICS()
    Do_Clustering.DBSCAN=Do_Clustering.perform_DBSCAN()

    for i, clusters in enumerate(Do_Clustering.K_Mean):
        Do_Table_Generate.new_table_generate(clusters, '../files/Clustering_adj/K_Means_outlier', file)
    Do_Table_Generate.new_table_generate(Do_Clustering.DBSCAN, '../files/Clustering_adj/DBSCAN', file)
    Do_Table_Generate.new_table_generate(Do_Clustering.Gaussian, '../files/Clustering_adj/Gaussian_Mixture_Model', file)
    Do_Table_Generate.new_table_generate(Do_Clustering.Agglomerative, '../files/Clustering_adj/Hierarchical_Agglomerative', file)
    Do_Table_Generate.new_table_generate(Do_Clustering.OPTIC, '../files/Clustering_adj/OPTICS', file)
    Do_Table_Generate.reversal_table_generate(data, '../files/Clustering_adj/Reversal', file)
