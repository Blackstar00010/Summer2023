from save_cointegration import *

# turn off warning
warnings.filterwarnings("ignore")

abnormal = False
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

min_max = False
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

    lst_mean = [0.299550651, 0.305943581, 0.289794729, 0.171507687, 0.299360048, 0.30047511, 0.314495301, 0.304478585,
                0.284678536, 0.296668338]
    lst_std = [0.106437896, 0.114507258, 0.114836032, 0.123466177, 0.115254803, 0.138503739, 0.135243158, 0.112620708,
               0.101942693, 0.1192455]

    for i in range(10):
        mean1 = lst_mean[i]
        std_dev1 = lst_std[i]
        sample_size1 = 384

        mean2 = 0.0643798208882231
        std_dev2 = 0.0643798208882231
        sample_size2 = 384

        # t-검정 수행
        t_statistic, p_value = stats.ttest_ind_from_stats(mean1, std_dev1, sample_size1, mean2, std_dev2, sample_size2)

        # 결과 출력
        print(f"{i}t-통계량:", t_statistic)
        print(f"{i}p-값:", p_value)

traded = False
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

finx = True
if finx:
    input_dir = '../finx/clustering_result'
    output_dir = '../finx/long_short_result'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0
    top_df = pd.DataFrame(columns=['month', 'invested', 'first', 'second', 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)

        # Call initial method
        Do_Result_Save = C.ResultCheck(data)

        # Do clustering and get 2D list of cluster index


        clusters = []
        for i in range(1 + max(set(data['Cluster Index']))):
            indices = list(data[data['Cluster Index'] == i].index)
            clusters.append(indices)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(clusters, output_dir, file, save=True, raw=True)

        outliers_count += (data['Cluster Index'] == 0).sum()/len(data)
        invested_num=len(data)-(data['Cluster Index'] == 0).sum()
        cl += len(clusters) - 1
        figure += Do_Result_Save.count_stock_of_traded()

        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in clusters]

            sublist_lengths = sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    'invested': invested_num,
                                    'first': [top3_lengths[0]],
                                    'second': [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)


    cl = int(cl / len(files))
    outliers_count = outliers_count / len(files)
    figure = figure / len(files)
    stat_list = [cl, 1 - outliers_count, outliers_count, figure]
    stat_df = pd.DataFrame(stat_list).T
    stat_df.columns=['Number of clusters', 'Number of stock in clusters',
                                    'Number of outliers', 'Number of stock traded']

    print(f'avg of clusters: {cl}')
    print(f'total outliers: {outliers_count}')
    print(f'number of stock traded: {figure}')

    top_df.to_csv(os.path.join('../finx/etc/', 'top3.csv'), index=False)

    stat_df.to_csv(os.path.join('../finx/etc/', 'cluster_info.csv'), index=False)

