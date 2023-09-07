from PCA_and_ETC import *

FTSE = True

if not FTSE:
    col = ['03/1990-12/1991', '01/1992-12/1994(Black_Wednesday)', '01/1995-12/2001(Dotcom_Bubble)', '01/2002-12/2006',
           '01/2007-12/2009(GFC)',
           '01/2010-12/2013(eurozone_crisis)', '01/2014-12/2015', '01/2016-12/2019(Brexit)',
           '01/2020-12/2022(Covid-19)', '01/2023-07/2023', 'Overall']

    period = [range(1, 23), range(23, 59), range(59, 143), range(143, 203), range(203, 239), range(239, 287),
              range(287, 311), range(311, 359), range(359, 395), range(395, 402), range(1, 402)]

if FTSE:
    col = ['03/1990-12/1991', '01/1992-12/1994(Black_Wednesday)', '01/1995-12/2001(Dotcom_Bubble)', '01/2002-12/2006',
           '01/2007-12/2009(GFC)',
           '01/2010-12/2013(eurozone_crisis)', '01/2014-12/2015', '01/2016-12/2019(Brexit)',
           '01/2020-12/2022(Covid-19)', 'Overall']

    period = [range(1, 23), range(23, 59), range(59, 143), range(143, 203), range(203, 239), range(239, 287),
              range(287, 311), range(311, 359), range(359, 395), range(1, 395)]

result = pd.read_csv('../files/result/total_result_modified.csv')

print('profit_factor')
profit_factor = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        pf = row[row > 0].sum() / np.abs(row[row < 0].sum())
        profit_factor.iloc[j, i] = pf

profit_factor['metric'] = 'Profit_factor'
print(profit_factor.to_string())

print('sharpe_ratio')
sharpe_ratio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        sf = (np.exp(row.mean() * 12) - 1) / (np.exp(row.std() * np.sqrt(12)) - 1)
        sharpe_ratio.iloc[j, i] = sf

sharpe_ratio['metric'] = 'Sharpe'
print(sharpe_ratio.to_string())

print('sortino_ratio')
sortino_ratio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        sf = (np.exp(row.mean() * 12) - 1) / (np.exp(row[row < 0].std() * np.sqrt(12)) - 1)
        sortino_ratio.iloc[j, i] = sf

sortino_ratio['metric'] = 'Sortino'
print(sortino_ratio.to_string())

print('Maximum_drawdown(MDD)')
MDD = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        row2=np.exp(row.astype(float))-1
        cumulative_returns=np.cumprod(1+row2)-1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak+1)
        max_drawdown = drawdown.min()
        MDD.iloc[j, i] = max_drawdown

MDD['metric'] = 'MDD'
print(MDD.to_string())

print('Calmar_ratio')
Calmar_ratio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        row2=np.exp(row.astype(float))-1
        cumulative_returns=np.cumprod(1+row2)-1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak+1)
        max_drawdown = drawdown.min()
        calmar = (np.exp(row.mean() * 12) - 1) / abs(max_drawdown)
        Calmar_ratio.iloc[j, i] = calmar

Calmar_ratio['metric'] = 'Calmar'
print(Calmar_ratio.to_string())

profit_factor = pd.concat([profit_factor, sharpe_ratio, sortino_ratio, MDD, Calmar_ratio], axis=0)
profit_factor.to_csv('../files/result/total.csv', index=True)
