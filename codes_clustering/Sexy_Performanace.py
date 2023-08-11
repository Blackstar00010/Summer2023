from PCA_and_ETC import *

base_directory = '../files/Clustering_adj_close/'

# Get all subdirectories in the base directory
subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

file_names = []
result_df = pd.DataFrame()

# Save subdir name in file_names at the beginning.
for subdir in subdirectories:
    # DBSCAN / Gaussian_Mixture_Model / HDBSCAN / Hierarchical_Agglomerative / ...
    file_names.append(subdir)

for subdir in subdirectories:
    # Long_Short_Merge.py
    directory = os.path.join(base_directory, subdir)
    long_short = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

    LS_merged_df = pd.DataFrame()

    for file in long_short:
        data = pd.read_csv(os.path.join(directory, file))

        # Keep only the 'Firm Name' and 'Long Short' columns
        data = data[['Firm Name', 'Long Short']]

        # Change the column name into file name (ex: 1990-01)
        file_column_name = os.path.splitext(file)[0]
        data = data.rename(columns={'Long Short': file_column_name})

        if LS_merged_df.empty:
            LS_merged_df = data
        else:
            LS_merged_df = pd.merge(LS_merged_df, data, on='Firm Name', how='outer')

    # Sort LS_Value according to Firm Name
    LS_merged_df = LS_merged_df.sort_values('Firm Name')

    '''ToDo: Firm Name이 중복되면 하나 drop. (K_mean_Outlier에 row 중복되는 것 있어서 오류 발생하여 추가)
    I don't know the reason why'''
    LS_merged_df = LS_merged_df.drop_duplicates(subset=LS_merged_df.columns[0], keep='first')

    # Set Firm Name column into index
    LS_merged_df.set_index('Firm Name', inplace=True)

    # 마지막 row 버리면 한칸씩 밀어버리는 것과 동치
    LS_merged_df = LS_merged_df.drop(LS_merged_df.columns[-1], axis=1)

    ###############################
    MOM_merged_df = pd.read_csv('../files/mom1_data_combined_adj_close.csv')

    # Set Firm Name column into index
    MOM_merged_df.set_index('Firm Name', inplace=True)

    # First row 버리고 LS_merged_df와 product
    # t-1 LS_Value와 t mom1 product
    MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)

    # ToDo: 혹시 몰라서 일단 NaN 0으로 대체. 없어도 될지도
    # MOM_merged_df = MOM_merged_df.fillna(0)
    LS_merged_df = LS_merged_df.fillna(0)

    # Multiply only the numeric columns
    prod = MOM_merged_df.values * LS_merged_df.values
    prod = pd.DataFrame(prod)

    # prod index set to df1.index
    prod.set_index(MOM_merged_df.index, inplace=True)
    # cumulative return은 1990-02부터 2022-12이기 때문에 prod.columns=df1.columns
    prod.columns = MOM_merged_df.columns

    # 제대로 됐나 확인하기 위해 csv saved.
    MOM_merged_df.to_csv('../files/adj_close/mom1.csv', index=True)
    LS_merged_df.to_csv(f'../files/adj_close/LS/{subdir}_LS.csv', index=True)
    prod.to_csv(f'../files/adj_close/prod/{subdir}_prod.csv', index=True)

    # Return_Check_Merge.py
    '''mom1과 LS_Value 곱한것 평균구하는 부분.
    Clustering/Result_Cheak_and_Save/LS_Table_Save 함수에서
    outlier cluster도 버리지 않는 대신 LS_Value=0으로 저장했기 때문에
    prod.mean 사용하면 안됨. prod에 모든 회사 row가 있기 때문에
    sum/(투자한 회사+투자안한 회사)로 계산되기 때문.'''
    # Count the non-zero LS that is the number of total firm invested(395 by 1 matrix/index=Date)
    non_zero_count = LS_merged_df.astype(bool).sum()
    # trade_firm_sum=0
    # for i in range(395):
    #     trade_firm_sum+=non_zero_count[i]
    # trade_firm_avg=trade_firm_sum/395
    # trade_firm_per=trade_firm_avg/811
    # print(subdir)
    # print(trade_firm_per)


    # sum about all rows(395 by 1 matrix/index=Date)
    column_sums = prod.sum()

    # calculate mean and make into DataFrame
    # column_means = column_sums / non_zero_count
    column_means = column_sums.values / non_zero_count.values
    column_means = pd.DataFrame(column_means)
    column_means.index = column_sums.index

    # Concat the means DataFrame to the result DataFrame(395 by 1 matrix->1 by 395 matrix)
    result_df = pd.concat([result_df, column_means.T], ignore_index=True)

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
file_names.append('FTSE 100')

# benchmark return merge with result_df
file = '../files/month_return.csv'
df = pd.read_csv(file)
df = df.iloc[1:]
df = df.iloc[0:, 85:]
print(df)
df.columns = result_df.columns  # columns name should be same with result_df
result_df = pd.concat([result_df, df], axis=0)  # add monthly_return right below result_df
result_df.index = file_names
result_df = result_df.astype(float)  # set data type as float(df.value was str actually.)
result_df = result_df.fillna(0)

# Save a new CSV file
result_df.to_csv('../files/result_adj_close.csv', index=True)

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

    plt.title('RETURN')
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
