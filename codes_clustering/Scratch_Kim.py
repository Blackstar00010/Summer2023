from PCA_and_ETC import *

# df=pd.read_csv('../files/mom1_data_combined_adj_close.csv')
#
# df=df.iloc[:, 1:]
# print(df)
#
# print(df.max().to_string())
#
# result=pd.DataFrame(df.max())
#
# result.to_csv('../files/max_mom1_before.csv')

from scipy import stats

# 예시 데이터 생성

lst_mean=[0.4647,1.0056,0.8436,1.2034,1.317,1.2209,1.1182,1.2699]
lst_std=[0.2296,0.3478,0.3172,0.3966,0.3527,0.4215,0.348,0.3467]

for i in range(8):
    mean1 = lst_mean[i]
    std_dev1 = lst_std[i]
    sample_size1 = 384

    mean2 = 0.2675
    std_dev2 = 0.1214
    sample_size2 = 384


    # t-검정 수행
    t_statistic, p_value = stats.ttest_ind_from_stats(mean1, std_dev1, sample_size1, mean2, std_dev2, sample_size2)

    # 결과 출력
    print(f"{i}t-통계량:", t_statistic)
    print(f"{i}p-값:", p_value)

