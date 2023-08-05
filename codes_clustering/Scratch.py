import warnings
import seaborn as sns
import Clustering as C
from Clustering import *
from PCA_and_ETC import *
from sklearn.datasets import load_iris

import os
import pandas as pd
import matplotlib.pyplot as plt

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

'''lab2 = True
if lab2:
    base_directory = '../files/Clustering_adj/'

    # Get all subdirectories in the base directory
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    file_names = []
    result_df = pd.DataFrame()
    for subdir in subdirectories:
        file_names.append(subdir)

    for subdir in subdirectories:
        directory = os.path.join(base_directory, subdir)
        long_short = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

        df2 = pd.DataFrame()

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

        df2.drop_duplicates(subset=df2.columns[0], keep='first', inplace=True)
        df2 = df2.sort_values('Firm Name')

        df1 = pd.read_csv('../files/mom1_data_combined_adj.csv')

        # Save the first column (index column)
        first_column = df2.iloc[:, 0]

        # 두 번째 열부터 마지막에서 두 번째 열까지 한 칸씩 오른쪽으로 영구적으로 이동시킵니다.
        for i in range(len(df2.columns) - 2, 0, -1):
            df2[df2.columns[i]] = df2[df2.columns[i - 1]]

        # 두 번째 열은 세 번째 열로 옮기고, 첫 번째 열은 그대로 둡니다.
        df2[df2.columns[1]] = df2[df2.columns[2]]

        # 마지막 열을 삭제합니다.
        df2.drop(df2.columns[-1], axis=1, inplace=True)

        df1.drop(df2.columns[1], axis=1, inplace=True)

        df1 = df1.fillna(0)
        df2 = df2.fillna(0)
        df2.columns = df1.columns

        # 각 열별로 0이 아닌 값들의 개수를 세어 저장할 딕셔너리를 생성합니다.
        non_zero_counts = {}

        # DataFrame의 열을 반복하면서 0이 아닌 값들의 개수를 세어 딕셔너리에 저장합니다.
        for column in df2.columns:
            non_zero_counts[column] = df2[column].astype(bool).sum()

        non_zero_counts.pop('Firm Name')
        # 딕셔너리를 DataFrame으로 변환합니다.
        non_zero_counts = pd.DataFrame(list(non_zero_counts.items()), columns=['Date', 'Value'])

        non_zero_counts.index=non_zero_counts.iloc[:,0]
        non_zero_counts=non_zero_counts.drop(non_zero_counts.columns[0], axis=1)
        non_zero_counts.reset_index(drop=True)



        # Multiply only the numeric columns
        prod = df1.iloc[:, 1:] * (df2.iloc[:, 1:])
        prod = pd.DataFrame(prod)

        prod.insert(0, 'Firm Name', first_column)



        df = prod.drop("Firm Name", axis=1)

        # Calculate the average of non-NaN values in each column (excluding the 'Firm Name' column)
        column_sums=df.sum()
        column_sums=pd.DataFrame(column_sums)

        column_means = column_sums.values/non_zero_counts.values
        column_means=pd.DataFrame(column_means)
        column_means.index=column_sums.index
        column_means=column_means.T


        # Concat the means DataFrame to the result DataFrame
        result_df = pd.concat([result_df, column_means], ignore_index=True)

    # Add a new column to the result DataFrame with the file names
    result_df.insert(0, 'Clustering Method', file_names)

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


    file = '../files/month_return.csv'
    df = pd.read_csv(file)
    df = df.iloc[1:]  # Jan data eliminate
    df = df.iloc[0:, 1:]  # save only data
    df.columns = result_df.columns  # columns name should be same with result_df
    result_df = pd.concat([result_df, df], axis=0)  # add monthly_return right below result_df
    result_df.index = file_names
    result_df = result_df.astype(float)  # set data type as float(df.value was str actually.)

    # # Save a new CSV file
    result_df = result_df.fillna(0)
    result_df.to_csv('../files/position_LS/result_ad.csv', index=True)

    # Add 1 to all data values
    result_df.iloc[:, 0:] = result_df.iloc[:, 0:] + 1

    # Calculate the cumulative product
    result_df.iloc[:, 0:] = result_df.iloc[:, 0:].cumprod(axis=1)

    # Subtract 1 to get back to the original scale
    result_df.iloc[:, 0:] = result_df.iloc[:, 0:] - 1

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
        plt.show()'''

# ToDo: BIRCH, Affinity Propagation
