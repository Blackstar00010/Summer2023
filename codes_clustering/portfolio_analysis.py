from PCA_and_ETC import *

col = ['02/1997-12/2001(DotCom_Bubble)', '01/2002-12/2006(Pax_americana)', '01/2007-12/2009(GFC)',
       '01/2010-12/2013(QE)',
       '01/2014-12/2017(eurozone_crisis)', '01/2018-12/2019(Brexit)', '01/2020-12/2022(corona)']

period = [range(1, 60), range(60, 120), range(120, 156), range(156, 204), range(204, 252), range(252, 276),
          range(276, 312)]

result = pd.read_csv('../files/result_adj_close.csv')
portfolio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
print(result.to_string())

for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        profit_factor = row[row > 0].sum() / np.abs(row[row < 0].sum())
        portfolio.iloc[j, i] = profit_factor

print(portfolio.to_string())

# # 연간 통계량 계산
# annual_return = df['Returns'].mean() * 252  # 252는 일반적인 거래일 수
# annual_stddev = df['Returns'].std() * np.sqrt(252)
#
# # 무위험 이자율 (예시)
# risk_free_rate = 0.02
#
# # 샤프 지수 계산
# sharpe_ratio = (annual_return - risk_free_rate) / annual_stddev
#
# # Downside Risk 계산 (Sortino Ratio를 위해)
# target_return = risk_free_rate  # 무위험 이자율을 타겟 리턴으로 설정
# negative_returns = df['Returns'][df['Returns'] < target_return]
# downside_stddev = negative_returns.std() * np.sqrt(252)
#
# # Sortino Ratio 계산
# sortino_ratio = (annual_return - risk_free_rate) / downside_stddev
#
# # Profit Factor 계산
# profit_factor = df['Returns'][df['Returns'] > 0].sum() / np.abs(df['Returns'][df['Returns'] < 0].sum())
#
# # Maximum Drawdown 계산
# cumulative_returns = (1 + df['Returns']).cumprod()
# peak = cumulative_returns.cummax()
# drawdown = (cumulative_returns - peak) / peak
# max_drawdown = drawdown.min()
#
# # Calmar Ratio 계산
# calmar_ratio = annual_return / np.abs(max_drawdown)
#
# print("Sharpe Ratio:", sharpe_ratio)
# print("Sortino Ratio:", sortino_ratio)
# print("Profit Factor:", profit_factor)
# print("Maximum Drawdown:", max_drawdown)
# print("Calmar Ratio:", calmar_ratio)
