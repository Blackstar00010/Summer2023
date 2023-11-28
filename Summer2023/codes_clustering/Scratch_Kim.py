import pandas as pd

from save_cointegration import *

# turn off warning
warnings.filterwarnings("ignore")

abnormal = True
if abnormal:
    MOM_merged_df = pd.read_csv('../files/mom1_data_combined_adj_close.csv')
    MOM_merged_df.set_index('Firm Name', inplace=True)
    MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)

    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0

    count3 += MOM_merged_df[MOM_merged_df >= 0.5].count().sum()
    count4 += MOM_merged_df[MOM_merged_df <= -0.5].count().sum()
    count5 += MOM_merged_df[MOM_merged_df == 0].count().sum()
    count6 += MOM_merged_df.isna().sum().sum()
    count_greater_than_0_5 = (MOM_merged_df >= 0.5).any(axis=1).sum()
    count_less_than_0_5 = (MOM_merged_df <= -0.5).any(axis=1).sum()
    count_both_0_5 = ((MOM_merged_df <= -0.5) | (MOM_merged_df >= 0.5)).any(axis=1).sum()


    print("-0.5보다 작고 0.5보다 큰 숫자가 있는 행의 개수:", count_both_0_5)
    print("0.5보다 큰 숫자가 있는 행의 개수:", count_greater_than_0_5)
    print("-0.5보다 작은 숫자가 있는 행의 개수:", count_less_than_0_5)
    print('0.5보다 큰 숫자가 있는 칸 갯수:', count3)
    print('-0.5보다 작은 숫자가 있는 칸 갯수:', count4)
    print('mom1=0인 칸 갯수:',count5)
    t=9749*391-count6
    print('NaN이 아닌 칸 갯수:', t)
    print(MOM_merged_df.shape)
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
        count1 = 0
        count2 = 0

        for i in range(len(LS_merged_df.iloc[0, :]) - 1):
            col = pd.DataFrame(LS_merged_df.iloc[:, i])

            firm_df = col.index[(col.iloc[:, 0] != 0) & (col.iloc[:, 0].notna())]

            value_df = MOM_merged_df.loc[firm_df]
            value_df = value_df.iloc[:, i]
            filtered_df = value_df[(value_df >= 0.5) | (value_df <= -0.5)]
            filtered_df = pd.DataFrame({filtered_df.name: filtered_df})

            count1 += value_df[value_df >= 0.5].count().sum()
            count2 += value_df[value_df <= -0.5].count().sum()

            outlier_df = pd.concat([outlier_df, filtered_df])

        outlier_df.to_csv(f'../files/abnormal_{subdir}.csv')

        count_numeric_df = pd.DataFrame(columns=['count'])
        # 각 열을 순회하면서 숫자가 있는 칸들의 개수를 계산하여 저장
        for col in outlier_df.columns:
            count = outlier_df[col].apply(lambda x: 1 if pd.notna(x) and isinstance(x, (int, float)) else 0).sum()
            count_numeric_df.loc[col] = [count]

        count_numeric_df.T.to_csv(f'../files/abnormal_count_{subdir}.csv')
        print(count1)
        print(count2)

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

finx_before = False
if finx_before:
    input_dir = '../finx/30_us_merged'
    # input_dir = '../finx/10_us_merged'
    output_dir = '../finx/clustering_result2'
    files = sorted(filename for filename in os.listdir(input_dir))
    for file in files:
        print(file)

        df = pd.read_csv(os.path.join(input_dir, file))

        new_df = pd.DataFrame(index=None, columns=['Firm Name', 'Momentum_1', 'Cluster Index'])
        new_df['Firm Name'] = df.iloc[:, 0]
        new_df['Momentum_1'] = df.iloc[:, 1]
        new_df['Cluster Index'] = df.iloc[:, 2] + 1

        new_df.to_csv(os.path.join(output_dir, file), index=None)

finx = False
if finx:
    input_dir = '../finx/clustering_result2'
    # output_dir = '../files/clustering_result/CL_10_1sigma'
    # output_dir = '../files/clustering_result/CL_10_2sigma'
    # output_dir = '../files/clustering_result/CL_30_1sigma'
    output_dir = '../files/clustering_result/CL_30_2sigma'

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

        outliers_count += (data['Cluster Index'] == 0).sum() / len(data)
        invested_num = len(data) - (data['Cluster Index'] == 0).sum()
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
    stat_df.columns = ['Number of clusters', 'Number of stock in clusters',
                       'Number of outliers', 'Number of stock traded']

    print(f'avg of clusters: {cl}')
    print(f'total outliers: {outliers_count}')
    print(f'number of stock traded: {figure}')

    top_df.to_csv(os.path.join('../finx/etc/', 'top3.csv'), index=False)

    stat_df.to_csv(os.path.join('../finx/etc/', 'cluster_info.csv'), index=False)
