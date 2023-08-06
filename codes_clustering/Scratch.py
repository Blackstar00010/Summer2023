import warnings
import seaborn as sns
import Clustering as C
from Clustering import *
from PCA_and_ETC import *
from sklearn.datasets import load_iris

# turn off warning
warnings.filterwarnings("ignore")

# sample data
iris = load_iris()
iris_pd = pd.DataFrame(iris.data[:, 2:], columns=['petal_length', 'petal_width'])

# Plot K_mean cluster about individual csv file
example1 = False
if example1:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([3])

    iris_pd['species'] = iris.target
    x_kc = Do_Clustering.test.cluster_centers_[:, 0]
    y_kc = Do_Clustering.test.cluster_centers_[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='petal_length', y='petal_width', hue='species', style='species', s=100, data=iris_pd)
    plt.scatter(x_kc, y_kc, s=100, color='r')
    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    plt.show()

    t_SNE('K_mean', Do_Clustering.PCA_Data, Do_Clustering.K_Mean_labels)

example2 = False
if example2:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN(0.9)

    n_clusters_ = len(set(Do_Clustering.DBSCAN_labels)) - (1 if -1 in Do_Clustering.DBSCAN_labels else 0)
    n_noise_ = list(Do_Clustering.DBSCAN_labels).count(-1)

    unique_labels = set(Do_Clustering.DBSCAN_labels)
    core_samples_mask = np.zeros_like(Do_Clustering.DBSCAN_labels, dtype=bool)
    core_samples_mask[Do_Clustering.test.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = Do_Clustering.DBSCAN_labels == k

        xy = Do_Clustering.PCA_Data[class_member_mask & core_samples_mask]
        plt.plot(
            xy.iloc[:, 0],
            xy.iloc[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = Do_Clustering.PCA_Data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy.iloc[:, 0],
            xy.iloc[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

    t_SNE('DBSCAN', Do_Clustering.PCA_Data, Do_Clustering.DBSCAN_labels)

example3 = False
if example3:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.Agglomerative = Do_Clustering.perform_HG(0.8)

    t_SNE('Hirarchical Agglormerative', Do_Clustering.PCA_Data, Do_Clustering.Agglomerative_labels)

example4 = False
if example4:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.Gaussian = Do_Clustering.perform_GMM(0.1)

    t_SNE('GMM', Do_Clustering.PCA_Data, Do_Clustering.Gaussian_labels)

example5 = False
if example5:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS(0.2)

    t_SNE('OPTICS', Do_Clustering.PCA_Data, Do_Clustering.OPTIC_labels)

example6 = False
if example6:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.menshift = Do_Clustering.perform_meanshift(0.3)

    t_SNE('meanshift', Do_Clustering.PCA_Data, Do_Clustering.menshift_labels)

example7 = False
if example7:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.HDBSCAN = Do_Clustering.perform_HDBSCAN(0.2)

    t_SNE('meanshift', Do_Clustering.PCA_Data, Do_Clustering.HDBSCAN_labels)

lab = False
if lab:
    input_dir = '../files/momentum_adj'
    files = sorted(filename for filename in os.listdir(input_dir))
    abnormal_file = []
    for file in files:
        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)

        if Do_Clustering.PCA_Data.shape[1] < 7:
            abnormal_file.append(file)

        t = find_optimal_GMM_hyperparameter(Do_Clustering.PCA_Data)
    print(abnormal_file)

calculate_and_plot_Return = False
if calculate_and_plot_Return:
    base_directory = '../files/Clustering_adj/'

    # Get all subdirectories in the base directory
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    file_names = []
    result_df = pd.DataFrame()

    # Save subdir name in file_names at the begining.
    for subdir in subdirectories:
        file_names.append(subdir)

    for subdir in subdirectories:
        directory = os.path.join(base_directory, subdir)
        long_short = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

        df2 = pd.DataFrame()

        # LS_Value 일단 이어 붙이기.
        for file in long_short:
            data = pd.read_csv(os.path.join(directory, file))

            # Keep only the 'Firm Name' and 'Long Short' columns
            data = data[['Firm Name', 'Long Short']]

            # Change the column name into file name (ex: 1990-01)
            file_column_name = os.path.splitext(file)[0]
            data = data.rename(columns={'Long Short': file_column_name})

            if df2.empty:
                df2 = data
            else:
                df2 = pd.merge(df2, data, on='Firm Name', how='outer')

        # Sort LS_Value according to Firm Name
        df2 = df2.sort_values('Firm Name')

        '''ToDo: Firm Name이 중복되면 하나 drop. (K_mean_Outlier에 row 중복되는 것 있어서 오류 발생하여 추가)
        I don't know the reason why'''
        df2 = df2.drop_duplicates(subset=df2.columns[0], keep='first')

        # Set Firm Name column into index
        df2.set_index('Firm Name', inplace=True)

        # 마지막 row 버리면 한칸씩 밀어버리는 것과 동치
        df2 = df2.drop(df2.columns[-1], axis=1)

        # read mom1_merge file
        df1 = pd.read_csv('../files/mom1_data_combined_adj.csv')

        # Set Firm Name column into index
        df1.set_index('Firm Name', inplace=True)

        # First row 버리고 df2와 product
        # t-1 LS_Value와 t mom1 product
        df1.drop(df1.columns[0], axis=1, inplace=True)

        # ToDo: 혹시 몰라서 일단 NaN 0으로 대체. 없어도 될지도
        df1 = df1.fillna(0)
        df2 = df2.fillna(0)

        # Multiply only the numeric columns
        prod = df1.values * df2.values
        prod = pd.DataFrame(prod)

        # prod index set to df1.index
        prod.set_index(df1.index, inplace=True)
        # cumulative return은 1990-02부터 2022-12이기 때문에 prod.columns=df1.columns
        prod.columns = df1.columns

        # 제대로 됐나 확인하기 위해 csv saved.
        # df1.to_csv('mom1.csv', index=True)
        # df2.to_csv(f'{subdir}_LS.csv', index=True)
        # prod.to_csv(f'{subdir}_prod.csv', index=True)

        '''mom1과 LS_Value 곱한것 평균구하는 부분.
        Clustering/Result_Cheak_and_Save/LS_Table_Save 함수에서
        outlier cluster도 버리지 않는 대신 LS_Value=0으로 저장했기 때문에
        prod.mean 사용하면 안됨. prod에 모든 회사 row가 있기 때문에
        sum/(투자한 회사+투자안한 회사)로 계산되기 때문.'''
        # Count the non-zero LS that is the number of total firm invested(395 by 1 matrix/index=Date)
        non_zero_count = df2.astype(bool).sum()

        # sum about all rows(395 by 1 matrix/index=Date)
        column_sums = prod.sum()

        # calculate mean and make into DataFrame
        column_means = column_sums.values / non_zero_count.values
        column_means = pd.DataFrame(column_means)
        column_means.index = column_sums.index

        # Concat the means DataFrame to the result DataFrame(395 by 1 matrix->1 by 395 matrix)
        result_df = pd.concat([result_df, column_means.T], ignore_index=True)

    # Add a new column to the result DataFrame with the file names
    result_df.insert(0, 'Clustering Method', file_names)

    # ToDo: 254부터 260 뭔지 모르지만 일단 나둠.
    # Separate the 'Clustering Method' column from the date columns
    clustering_method = result_df['Clustering Method']
    date_columns_df = result_df.drop('Clustering Method', axis=1)

    # Convert the date columns to datetime format and sort them
    date_columns_df.columns = pd.to_datetime(date_columns_df.columns, errors='coerce')
    date_columns_df = date_columns_df.sort_index(axis=1)

    # Concat the 'Clustering Method' column back with the sorted date columns
    result_df = pd.concat([clustering_method, date_columns_df], axis=1)
    result_df.set_index('Clustering Method', inplace=True)
    file_names.append('Benchmark')

    # benchmark return merge with result_df
    file = '../files/month_return.csv'
    df = pd.read_csv(file)
    df = df.iloc[1:]  # Jan data eliminate
    df = df.iloc[0:, 1:]  # save only data
    df.columns = result_df.columns  # columns name should be same with result_df
    result_df = pd.concat([result_df, df], axis=0)  # add monthly_return right below result_df
    result_df.index = file_names
    result_df = result_df.astype(float)  # set data type as float(df.value was str actually.)
    result_df = result_df.fillna(0) # 혹시 몰라서

    # Save a new CSV file
    # result_df.to_csv('Scratch_Files/result.csv', index=True)

    # Add 1 to all data values
    result_df.iloc[:, 0:] = result_df.iloc[:, 0:] + 1

    # Calculate the cumulative product
    result_df.iloc[:, 0:] = result_df.iloc[:, 0:].cumprod(axis=1)

    # Subtract 1 to get back to the original scale
    result_df.iloc[:, 0:] = result_df.iloc[:, 0:] - 1

    Plot = True
    if Plot:
        plt.figure(figsize=(10, 6))

        for i in range(len(result_df)):
            plt.plot(result_df.columns[1:], result_df.iloc[i, 1:], label=result_df.iloc[i, 0])

        plt.title('Average Values Over Time')
        plt.xlabel('Date')
        plt.ylabel('cumulative Value')
        plt.xticks(rotation=45)
        plt.legend(result_df.index)  # Add a legend to distinguish different lines
        plt.tight_layout()
        plt.show()

        # Plot a graph for each row
        for i in range(len(result_df)):
            plt.figure(figsize=(10, 6))
            plt.plot(result_df.columns[1:], result_df.iloc[i, 1:])
            plt.title(result_df.index[i])
            plt.xlabel('Date')
            plt.ylabel('Average Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# ToDo: BIRCH, Affinity Propagation
