import Clustering as C
from PCA_and_ETC import *

MOM_merged_df = pd.read_csv('../files/mom1_data_combined_adj_close.csv')
MOM_merged_df.set_index('Firm Name', inplace=True)
MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)

# Save K_mean clutering method LS_Tables
# hyper parameter K(2,3,4,5,10,50,100,500,1000) should be tested manually.(paper follow)
K_mean_Save = True
if K_mean_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/K_Means_outlier'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in [100]:
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
            Do_Clustering.perform_kmeans(i, 0.5)

            Do_Result_Save.ls_table(Do_Clustering.K_Mean[0], output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.K_Mean_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.K_Mean[0])
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.K_Mean_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result('K_mean', result_df, file_names, FTSE=True, apply_log=True)
    save_cluster_info('K_mean', stat_lists, file_names)

# Save DBSCAN clutering method LS_Tables
# hyper parameter eps percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow)
dbscan_Save = False
if dbscan_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/DBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in np.arange(0.9, 1, 0.01):
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
            Do_Clustering.perform_DBSCAN(i,True)

            Do_Result_Save.ls_table(Do_Clustering.DBSCAN, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.DBSCAN_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.DBSCAN_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result('DBSCAN', result_df, file_names, FTSE=True, apply_log=True)
    save_cluster_info('DBSCAN', stat_lists, file_names)

# Save HDBSCAN clutering method LS_Tables
# hyper parameter distance percentile np.range(0.1, 1, 0.1) should be tested manually.
hdbscan_Save = False
if hdbscan_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/HDBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

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
            Do_Clustering.perform_HDBSCAN(i)

            Do_Result_Save.ls_table(Do_Clustering.HDBSCAN, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.HDBSCAN_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.HDBSCAN_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result('HDBSCAN', result_df, file_names, FTSE=True, apply_log=True)
    save_cluster_info('HDBSCAN', stat_lists, file_names)

# Save GaussianMixture clutering method LS_Tables
# hyper parameter outlier probability [40,35,30,25,20,15,10,5,1] should be tested manually.(paper follow)
GMM_Save = False
if GMM_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/GMM'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

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
            Do_Clustering.perform_GMM(i)

            Do_Result_Save.ls_table(Do_Clustering.Gaussian, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.Gaussian_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.Gaussian_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result('GMM', result_df, file_names, FTSE=True, apply_log=True)
    save_cluster_info('GMM', stat_lists, file_names)

# Save Hirarchical Agglomerative clutering method LS_Tables
# hyper parameter distance percentile np.range(0.1, 1, 0.1) should be tested manually.
agglomerative_Save = False
if agglomerative_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Hierarchical_Agglomerative'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

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
            Do_Clustering.perform_HA(i)

            Do_Result_Save.ls_table(Do_Clustering.Agglomerative, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.Agglomerative_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Agglomerative)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.Agglomerative_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result('Agglomerative', result_df, file_names, FTSE=True, apply_log=True)
    save_cluster_info('Agglomerative', stat_lists, file_names)

# Save OPTICS clutering method LS_Tables
# hyper parameter percentile of xi [0.01,0.03,0.05,0.07,0.09,0.1,0.3,0.5,0.7] should be tested manually.
optics_Save = False
if optics_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/OPTICS'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

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
            Do_Clustering.perform_OPTICS(i)

            Do_Result_Save.ls_table(Do_Clustering.OPTIC, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.OPTIC_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.OPTIC_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result('OPTICS', result_df, file_names, FTSE=True, apply_log=True)
    save_cluster_info('OPTICS', stat_lists, file_names)

# Save Meanshift clutering method LS_Tables
# hyper parameter quantile np.range(0.1, 1, 0.1) should be tested manually.(paper follow)
meanshift_Save = False
if meanshift_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/meanshift'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

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
            Do_Clustering.perform_meanshift(i)

            Do_Result_Save.ls_table(Do_Clustering.meanshift, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.meanshift_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.meanshift)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.meanshift_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result('meanshift', result_df, file_names, FTSE=True, apply_log=True)
    save_cluster_info('meanshift', stat_lists, file_names)

# Save BIRCH clutering method LS_Tables
# hyper parameter percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow)
birch_Save = False
if birch_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/BIRCH'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

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
            Do_Clustering.perform_BIRCH(i)

            Do_Result_Save.ls_table(Do_Clustering.BIRCH, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.BIRCH_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.BIRCH)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.BIRCH_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result('BIRCH', result_df, file_names, FTSE=True, apply_log=True)
    save_cluster_info('BIRCH', stat_lists, file_names)
