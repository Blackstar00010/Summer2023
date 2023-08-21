from PCA_and_ETC import *

col = ['02/1997-12/2001(DotCom_Bubble)', '01/2002-12/2006(Pax_americana)', '01/2007-12/2009(GFC)',
       '01/2010-12/2013(QE)',
       '01/2014-12/2017(eurozone_crisis)', '01/2018-12/2019(Brexit)', '01/2020-12/2022(corona)']

period = [range(1, 60), range(60, 120), range(120, 156), range(156, 204), range(204, 252), range(252, 276),
          range(276, 312)]

result = pd.read_csv('../files/result/result_modified.csv')

print('profit_factor')
profit_factor = pd.DataFrame(index=col, columns=result.iloc[:, 0])

for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        pf = row[row > 0].sum() / np.abs(row[row < 0].sum())
        profit_factor.iloc[j, i] = pf

print(profit_factor.to_string())

print('sharpe_ratio')
sharpe_ratio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        sf = row.mean() / row.std()
        sharpe_ratio.iloc[j, i] = sf

print(sharpe_ratio.to_string())

print('sortino_ratio')
sortino_ratio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        sf = row.mean() / row[row<0].std()
        sortino_ratio.iloc[j, i] = sf

print(sortino_ratio.to_string())

print('Maximum_drawdown(MDD)')
MDD = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        cumulative_returns = (1 + row).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        MDD.iloc[j, i] = max_drawdown

print(MDD.to_string())

print('Calmar_ratio')
Calmar_ratio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        cumulative_returns = (1 + row).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        calmar=row.mean()/max_drawdown
        Calmar_ratio.iloc[j, i] = calmar

print(Calmar_ratio.to_string())

base_directory = '../files/Clustering_adj_close/'

# Get all subdirectories in the base directory
subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
subdirectories.remove('Reversal')
for subdir in subdirectories:
    print(subdir)
    # Long_Short_Merge.py
    directory = os.path.join(base_directory, subdir)
    long_short = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))


    LS_merged_df = pd.DataFrame()

    for file in long_short:
        data = pd.read_csv(os.path.join(directory, file))

        # Keep only the 'Firm Name' and 'Long Short' columns
        data = data[['Firm Name', 'Cluster Index']]

        # Change the column name into file name (ex: 1990-01)
        file_column_name = os.path.splitext(file)[0]
        data = data.rename(columns={'Cluster Index': file_column_name})


        if LS_merged_df.empty:
            LS_merged_df = data
        else:
            LS_merged_df = pd.merge(LS_merged_df, data, on='Firm Name', how='outer')

    # Set Firm Name column into index
    LS_merged_df.set_index('Firm Name', inplace=True)

    # Sort LS_Value according to Firm Name
    LS_merged_df = LS_merged_df.max().mean()

    print(LS_merged_df)