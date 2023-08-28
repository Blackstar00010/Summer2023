import Clustering as C
from PCA_and_ETC import *

MOM_merged_df = pd.read_csv('../files/mom1_data_combined_adj_close.csv')
# Set Firm Name column into index
MOM_merged_df.set_index('Firm Name', inplace=True)
MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)

# Save K_mean clutering method LS_Tables
# hyper parameter K(2,3,4,5,10,50,100,500,1000) should be tested manually.(paper follow)
K_mean_Save = False
if K_mean_Save:
    file_names = []
    result_df = pd.DataFrame()

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/K_Means_outlier'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0

    for i in [2, 3, 4, 5, 10, 50, 100, 500, 1000]:
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

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
            Do_Clustering.K_Mean = Do_Clustering.perform_kmeans(i, 0.5)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.K_Mean[0])

            table = Do_Result_Save.ls_table(Do_Clustering.K_Mean, output_dir, file, save=False, raw=False)

            LS_merged_df = merge_LS_Table(table, LS_merged_df, file)

            cl += len(sorted(list(set(Do_Clustering.K_Mean_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.K_Mean_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)

        cl = cl / len(files)
        print('average number of clusters:', cl)
        print(f'total outliers: {outliers_count}')

    print(result_df)
    save_and_plot_LS_Table(result_df, file_names, FTSE=True, apply_log=True)

# Save DBSCAN clutering method LS_Tables
# hyper parameter eps percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow)
dbscan_Save = False
if dbscan_Save:
    file_names = []
    result_df = pd.DataFrame()

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/DBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0

    for i in np.arange(0.1, 1, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

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
            Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(i)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)

            table = Do_Result_Save.ls_table(Do_Clustering.DBSCAN, output_dir, file, save=False, raw=False)

            LS_merged_df = merge_LS_Table(table, LS_merged_df, file)

            cl += len(sorted(list(set(Do_Clustering.DBSCAN_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.DBSCAN_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)

        cl = cl / len(files)
        print('average number of clusters:', cl)
        print(f'total outliers: {outliers_count}')

    print(result_df)
    save_and_plot_LS_Table(result_df, file_names, FTSE=True, apply_log=True)

# Save HDBSCAN clutering method LS_Tables
# hyper parameter distance percentile np.range(0.1, 1, 0.1) should be tested manually.
hdbscan_Save = False
if hdbscan_Save:
    file_names = []
    result_df = pd.DataFrame()

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/HDBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0

    for i in np.arange(0.1, 1, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

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
            Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN(i)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

            table = Do_Result_Save.ls_table(Do_Clustering.HDBSCAN, output_dir, file, save=False, raw=False)

            LS_merged_df = merge_LS_Table(table, LS_merged_df, file)

            cl += len(sorted(list(set(Do_Clustering.HDBSCAN_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.HDBSCAN_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)

        cl = cl / len(files)
        print('average number of clusters:', cl)
        print(f'total outliers: {outliers_count}')

    print(result_df)
    save_and_plot_LS_Table(result_df, file_names, FTSE=True, apply_log=True)

# Save GaussianMixture clutering method LS_Tables
# hyper parameter outlier probability [40,35,30,25,20,15,10,5,1] should be tested manually.(paper follow)
GMM_Save = False
if GMM_Save:
    file_names = []
    result_df = pd.DataFrame()

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Gaussian'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0

    for i in [40, 35, 30, 25, 20, 15, 10, 5, 1]:
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

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
            Do_Clustering.Gaussian = Do_Clustering.perform_GMM(i)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)

            table = Do_Result_Save.ls_table(Do_Clustering.Gaussian, output_dir, file, save=False, raw=False)

            LS_merged_df = merge_LS_Table(table, LS_merged_df, file)

            cl += len(sorted(list(set(Do_Clustering.Gaussian_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.Gaussian_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)

        cl = cl / len(files)
        print('average number of clusters:', cl)
        print(f'total outliers: {outliers_count}')

    print(result_df)
    save_and_plot_LS_Table(result_df, file_names, FTSE=True, apply_log=True)

# Save Hirarchical Agglomerative clutering method LS_Tables
# hyper parameter distance percentile np.range(0.1, 1, 0.1) should be tested manually.
Agglormerative_Save = False
if Agglormerative_Save:
    file_names = []
    result_df = pd.DataFrame()

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Hierarchical_Agglomerative'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0

    for i in np.arange(0.1, 0.3, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

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

            table = Do_Result_Save.ls_table(Do_Clustering.Agglomerative, output_dir, file, save=False, raw=False)

            LS_merged_df = merge_LS_Table(table, LS_merged_df, file)

            cl += len(sorted(list(set(Do_Clustering.Agglomerative_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.Agglomerative_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)

        cl = cl / len(files)
        print('average number of clusters:', cl)
        print(f'total outliers: {outliers_count}')
    print(result_df)

    save_and_plot_LS_Table(result_df, file_names, FTSE=True, apply_log=True)

# Save OPTICS clutering method LS_Tables
# hyper parameter percentile of xi [0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5,0.7] should be tested manually.
optics_Save = False
if optics_Save:
    file_names = []
    result_df = pd.DataFrame()

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/OPTICS'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0

    for i in [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7]:
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

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
            Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(i)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)

            table = Do_Result_Save.ls_table(Do_Clustering.OPTIC, output_dir, file, save=False, raw=False)

            LS_merged_df = merge_LS_Table(table, LS_merged_df, file)

            cl += len(sorted(list(set(Do_Clustering.OPTIC_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.OPTIC_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)

        cl = cl / len(files)
        print('average number of clusters:', cl)
        print(f'total outliers: {outliers_count}')

    print(result_df)
    save_and_plot_LS_Table(result_df, file_names, FTSE=True, apply_log=True)

# Save Meanshift clutering method LS_Tables
# hyper parameter quantile np.range(0.1, 1, 0.1) should be tested manually.(paper follow)
meanshift_Save = False
if meanshift_Save:
    file_names = []
    result_df = pd.DataFrame()

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/meanshift'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0

    for i in np.arange(0.1, 1, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

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
            Do_Clustering.meanshift = Do_Clustering.perform_meanshift(i)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.meanshift)

            table = Do_Result_Save.ls_table(Do_Clustering.meanshift, output_dir, file, save=False, raw=False)

            LS_merged_df = merge_LS_Table(table, LS_merged_df, file)

            cl += len(sorted(list(set(Do_Clustering.meanshift_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.meanshift_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)

        cl = cl / len(files)
        print('average number of clusters:', cl)
        print(f'total outliers: {outliers_count}')

    print(result_df)
    save_and_plot_LS_Table(result_df, file_names, FTSE=True, apply_log=True)

# Save BIRCH clutering method LS_Tables
# hyper parameter percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow)
birch_Save = True
if birch_Save:
    file_names = []
    result_df = pd.DataFrame()

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/BIRCH'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    cl = 0

    for i in np.arange(0.1, 0.3, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

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
            Do_Clustering.BIRCH = Do_Clustering.perform_BIRCH(i)

            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.BIRCH)

            table = Do_Result_Save.ls_table(Do_Clustering.BIRCH, output_dir, file, save=False, raw=False)

            LS_merged_df = merge_LS_Table(table, LS_merged_df, file)

            cl += len(sorted(list(set(Do_Clustering.BIRCH_labels))))
            print("Number of clusters is:", len(set(Do_Clustering.BIRCH_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)

        cl = cl / len(files)
        print('average number of clusters:', cl)
        print(f'total outliers: {outliers_count}')

    print(result_df)
    save_and_plot_LS_Table(result_df, file_names, FTSE=True, apply_log=True)
