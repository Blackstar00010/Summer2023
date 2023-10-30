import pandas as pd

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




def product_LS_Table(LS_merged_df: pd.DataFrame, MOM_merged_df: pd.DataFrame, result_df: pd.DataFrame, subdir, save=False):
    # Sort LS_Value according to Firm Name
    LS_merged_df = LS_merged_df.sort_values('Firm Name')

    # Set Firm Name column into index
    LS_merged_df.set_index('Firm Name', inplace=True)

    # 마지막 row 버리면 한칸씩 밀어버리는 것과 동치
    LS_merged_df = LS_merged_df.drop(LS_merged_df.columns[-1], axis=1)
    LS_merged_df = LS_merged_df.fillna(0)
    LS_merged_df.columns = MOM_merged_df.columns

    if save:
        LS_merged_df.to_csv(f'../files/LS_merge_{subdir}.csv',index=False)

    prod = MOM_merged_df * LS_merged_df
    prod = pd.DataFrame(prod)

    # prod index set to df1.index
    prod.set_index(MOM_merged_df.index, inplace=True)
    # cumulative return은 1990-02부터 2022-12이기 때문에 prod.columns=df1.columns
    prod.columns = MOM_merged_df.columns

    if save:
        prod.to_csv(f'../files/prod_{subdir}.csv')

    if True:
        for col in prod.columns:
                if col == 'Firm Name':
                    continue
                prod.loc[prod[col] > 0.5, col] = 0.5
                prod.loc[prod[col] < -0.33333, col] = -0.33333

    # Count the non-zero LS that is the number of total firm invested(395 by 1 matrix/index=Date)
    non_zero_count = LS_merged_df.astype(bool).sum()

    non_zero_count2=pd.DataFrame(non_zero_count).T
    # non_zero_count2.to_csv(f'../files/traded_{subdir}.csv')


    # sum about all rows(395 by 1 matrix/index=Date)
    column_sums = prod.sum()

    # calculate mean and make into DataFrame
    column_means = column_sums.values / non_zero_count.values
    column_means = pd.DataFrame(column_means)
    column_means.index = column_sums.index

    # Concat the means DataFrame to the result DataFrame(395 by 1 matrix->1 by 395 matrix)
    result_df = pd.concat([result_df, column_means.T], ignore_index=True)

    return result_df


def save_and_plot_result(clustering_name, result_df: pd.DataFrame, file_names, FTSE=False, apply_log=True, new_Plot=False):
    # Add a new column to the result DataFrame with the file names
    result_df['Clustering Method'] = file_names

    # Separate the 'Clustering Method' column from the date columns
    clustering_method = result_df['Clustering Method']
    date_columns_df = result_df.drop('Clustering Method', axis=1)

    # Convert the date columns to datetime format and sort them
    date_columns_df.columns = pd.to_datetime(date_columns_df.columns, errors='coerce')
    date_columns_df = date_columns_df.sort_index(axis=1)

    # Concat the 'Clustering Method' column back with the sorted date columns
    result_df = pd.concat([clustering_method, date_columns_df], axis=1)
    result_df.set_index('Clustering Method', inplace=True)

    if FTSE:
        file_names.append('FTSE 100')
        file = '../files/ftse_return.csv'
        df = pd.read_csv(file)
        df = df.iloc[1:]
        df = df.iloc[0:, 2:]
        df.columns = result_df.columns[0:-7]  # columns name should be same with result_df
        result_df = pd.concat([result_df.iloc[:, 0:-7], df], axis=0)  # add monthly_return right below result_df

    result_df.index = file_names
    result_df = result_df.astype(float)  # set data type as float(df.value was str actually.)
    result_df = result_df.fillna(0)

    # Add 1 to all data values
    result_df.iloc[:, 0:] = result_df.iloc[:, 0:] + 1

    # transform into log scale
    result_df.iloc[:, 0:] = np.log(result_df.iloc[:, 0:]) if apply_log else result_df.iloc[:, 0:]
    result_df.to_csv(os.path.join('../files/result/', f'{clustering_name}_result_original.csv'), index=True)

    result_modified = pd.DataFrame(
        index=['count', 'annual return mean', 'annual return std', 'monthly return min', 'monthly return max'],
        columns=result_df.index)
    for i in range(len(result_modified.columns)):
        result_modified.iloc[0, i] = len(result_df.columns)
        result_modified.iloc[1, i] = np.exp(np.mean(result_df.iloc[i, :]) * 12) - 1
        result_modified.iloc[2, i] = np.exp(np.std(result_df.iloc[i, :]) * np.sqrt(12)) - 1
        result_modified.iloc[3, i] = np.min(result_df.iloc[i, :])
        result_modified.iloc[4, i] = np.max(result_df.iloc[i, :])
    # result_modified.iloc[1, :] = result_modified.iloc[1, :] * len(result_df.iloc[1, :]) / 12

    sharpe_ratio = pd.DataFrame(index=['Sharpe ratio'], columns=result_modified.columns)
    for i in range(len(result_modified.columns)):
        sharpe_ratio.iloc[0, i] = result_modified.iloc[1, i] / result_modified.iloc[2, i]

    result_modified = pd.concat([result_modified, sharpe_ratio], axis=0)

    MDD = pd.DataFrame(index=['Maximum drawdown'], columns=result_modified.columns)
    for i in range(len(result_df.index)):
        row = result_df.iloc[i, :]
        row2 = np.exp(row.astype(float)) - 1
        cumulative_returns = np.cumprod(1 + row2) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1)
        max_drawdown = drawdown.min()
        MDD.iloc[0, i] = max_drawdown

    result_modified = pd.concat([result_modified, MDD], axis=0)

    result_modified.to_csv(os.path.join('../files/result/', clustering_name + '_statistcs_modified.csv'), index=True)
    result_df.to_csv(os.path.join('../files/result/', clustering_name + '_result_modified.csv'), index=True)