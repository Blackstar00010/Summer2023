from PCA_and_ETC import *

# turn off warning
warnings.filterwarnings("ignore")

abnormal = True
if abnormal:
    MOM_merged_df = pd.read_csv('../files/mom1_data_combined_adj_close.csv')
    MOM_merged_df.set_index('Firm Name', inplace=True)
    MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)

    base_directory = '../files/clustering_result/'

    # Get all subdirectories in the base directory
    subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

    file_names = []
    result_df = pd.DataFrame()

    for subdir in subdirectories:
        print(subdir)
        directory = os.path.join(base_directory, subdir)

        LS_merged_df = pd.DataFrame()

        files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))
        for file in files:
            data = pd.read_csv(os.path.join(directory, file))
            LS_merged_df = merge_LS_Table(data, LS_merged_df, file)

        LS_merged_df.index = LS_merged_df.iloc[:, 0]
        LS_merged_df.drop(columns='Firm Name', inplace=True)

        outlier_df = pd.DataFrame()

        for i in range(len(LS_merged_df.iloc[0, :]) - 1):
            col = pd.DataFrame(LS_merged_df.iloc[:, i])

            firm_df = col.index[(col.iloc[:, 0] != 0) & (col.iloc[:, 0].notna())]

            value_df = MOM_merged_df.loc[firm_df]
            value_df = value_df.iloc[:, i]
            filtered_df = value_df[(value_df >= 1) | (value_df <= -0.5)]
            filtered_df = pd.DataFrame({filtered_df.name: filtered_df})

            outlier_df = pd.concat([outlier_df, filtered_df])

        outlier_df.to_csv(f'../files/abnormal_{subdir}.csv')

        count_numeric_df = pd.DataFrame(columns=['count'])
        # 각 열을 순회하면서 숫자가 있는 칸들의 개수를 계산하여 저장
        for col in outlier_df.columns:
            count = outlier_df[col].apply(lambda x: 1 if pd.notna(x) and isinstance(x, (int, float)) else 0).sum()
            count_numeric_df.loc[col] = [count]

        count_numeric_df.T.to_csv(f'../files/abnormal_count_{subdir}.csv')

min_max = True
if min_max:
    df = pd.read_csv('../files/mom1_data_combined_adj_close.csv')

    df = df.iloc[:, 1:]
    print(df)

    print(df.max().to_string())

    result = pd.DataFrame(df.max())

    result.to_csv('../files/max_mom1.csv')

    result = pd.DataFrame(df.min())

    result.to_csv('../files/min_mom1.csv')

t_test = False
if t_test:
    from scipy import stats

    lst_mean = [0.1265, 0.1274, 0.1356, 0.1271, 0.1164, 0.1107, 0.1053, 0.1188]
    lst_std = [0.102, 0.1048, 0.1063, 0.1049, 0.1021, 0.0917, 0.1268, 0.0864]

    for i in range(8):
        mean1 = lst_mean[i]
        std_dev1 = lst_std[i]
        sample_size1 = 384

        mean2 = 0.0856
        std_dev2 = 0.0827
        sample_size2 = 384

        # t-검정 수행
        t_statistic, p_value = stats.ttest_ind_from_stats(mean1, std_dev1, sample_size1, mean2, std_dev2, sample_size2)

        # 결과 출력
        print(f"{i}t-통계량:", t_statistic)
        print(f"{i}p-값:", p_value)

traded = True
if traded:
    traded_df = pd.read_csv('../files/traded_DBSCAN.csv')
    traded_df.columns = pd.to_datetime(traded_df.columns, errors='coerce')

    traded_df.T.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('Date')
    plt.ylabel('Number of stock traded')
    plt.title('Stock Traded Per Month')
    plt.grid(axis='y')
    plt.xticks([])
    plt.legend().set_visible(False)
    plt.show()
