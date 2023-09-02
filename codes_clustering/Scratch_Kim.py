import os
from PCA_and_ETC import *

input_dir = '../files/characteristics'
output_dir = '../files/clustering_result/Cointegration'

files = sorted(filename for filename in os.listdir(input_dir))
is_jamesd = 'jamesd' in os.path.abspath('.')
for file in files:
    print(file)
    if file in os.listdir(output_dir):
        continue
    data = read_and_preprocess_data(input_dir, file)

    mom_data = read_mom_data(data)

    # inv_list = find_cointegrated_pairs_deprecated(mom_data)
    inv_list = find_cointegrated_pairs(mom_data)

    LS_Table = True
    if LS_Table:
        save_cointegrated_LS(output_dir, file, mom_data, inv_list)

if True:
    def read_mom_data(data):
        # mom1 save and data Normalization
        mom1 = data.values.astype(float)[:, 0]
        data_normalized = (data - data.mean()) / data.std()
        mat = data_normalized.values.astype(float)

        # mom1을 제외한 mat/PCA(2-49)
        # mat = np.delete(mat, 0, axis=1)

        # mom49를 제외한 mat/PCA(1-48)
        mat = np.delete(mat, 48, axis=1)

        df_combined = pd.DataFrame(mat)
        df_combined.insert(0, 'Mom1', mom1)
        df_combined.index = data.index

        return df_combined.T


    def cointegrate(data, s1, s2):
        x = data[s1].values
        y = data[s2].values
        _, p_value, _ = coint(x, y)
        return p_value


    def adf_result(pair):
        try:
            ret = sm.tsa.adfuller(pair)[1]
        except ValueError:
            ret = 0.06
        return ret


    def kpss_result(pair):
        try:
            ret = kpss(pair)[1]
        except:
            ret = 0.04
        return ret


    def find_cointegrated_pairs(data: pd.DataFrame) -> list:
        n_jobs = 8
        data = data.iloc[1:, :]
        pairs = pd.DataFrame(combinations(data.columns, 2))  # 모든 회사 조합
        pairs['pvalue'] = Parallel(n_jobs=n_jobs)(delayed(cointegrate)(data, pair[0], pair[1]) for pair in pairs.values)
        pairs = pairs.loc[pairs.index[pairs['pvalue'] < 0.01], :]
        print('Finished filtering pairs using pvalue!')

        spread_df: pd.DataFrame = pairs.apply(lambda x: data[x[0]] - data[x[1]], axis=1)
        # spread_df['adf_result'] = Parallel(n_jobs=n_jobs)(delayed(adf_result)(pair) for pair in spread_df.values)
        # spread_df = spread_df.loc[spread_df.index[spread_df['adf_result'] < 0.05], :]
        # print('Finished filtering pairs using adf_result!')

        spread_df['kpss_result'] = Parallel(n_jobs=n_jobs)(delayed(kpss_result)(pair) for pair in spread_df.values)
        spread_df = spread_df.loc[spread_df.index[spread_df['kpss_result'] > 0.05], :]
        spread_df = spread_df.drop(columns=['adf_result'])
        spread_df = spread_df.drop(columns=['kpss_result'])
        print('Finished filtering pairs using kpss_result!')

        spread_sr = spread_df.iloc[:, 0]
        pairs['spread'] = (spread_sr - spread_sr.mean()) / spread_sr.std()
        pairs = pairs.dropna(subset=['spread'])
        pairs = pd.DataFrame(pairs)
        pairs = pairs.loc[pairs.index[pairs['spread'].abs() > 2], :]
        pairs['pair1'] = pairs[0] * (pairs['spread'] > 0) + pairs[1] * (pairs['spread'] <= 0)
        pairs['pair2'] = pairs[0] * (pairs['spread'] <= 0) + pairs[1] * (pairs['spread'] > 0)
        pairs = pairs.drop(columns=[0, 1])
        print('Finished filtering pairs using normalised spread!')

        pairs.sort_values(by='pvalue', axis=0, inplace=True)
        pairs = pairs.drop_duplicates(subset='pair1')
        pairs = pairs.drop_duplicates(subset='pair2')
        invest_list = pd.DataFrame(pairs.values.tolist())
        invest_list = invest_list.iloc[:, 2:4]
        invest_list = invest_list.values.tolist()

        return invest_list


    def find_cointegrated_pairs_deprecated(data: pd.DataFrame):
        """
        Deprecated
        :param data:
        :return:
        """
        data = data.iloc[1:, :]
        invest_list = []

        pairs = list(combinations(data.columns, 2))  # 모든 회사 조합
        print(len(pairs))
        pairs_len = 1

        count_p = 0
        count_s = 0
        count_n = 0

        while len(pairs) != pairs_len:
            pairs_len = len(pairs)

            for i, pair in enumerate(pairs):
                pvalue = cointegrate(data, pair[0], pair[1])

                if pvalue > 0.01:
                    continue

                else:
                    count_p += 1
                    spread = data[pair[0]] - data[pair[1]]
                    adf_result = sm.tsa.adfuller(spread)
                    kpss_result = kpss(spread)

                    if adf_result[1] > 0.05 or kpss_result[1] < 0.05:
                        continue

                    else:
                        count_s += 1
                        mean_spread = spread.mean()
                        std_spread = spread.std()
                        z_score = (spread - mean_spread) / std_spread

                        spread_value = float(z_score[0])

                        if abs(spread_value) <= 2:
                            continue


                        elif spread_value < -2:
                            pair = (pair[1], pair[0])
                            # pair = (pair[1], pair[0], pvalue, adf_result[1], kpss_result[1], spread_value)
                            invest_list.append(pair)
                            pairs = [p for p in pairs if all(item not in pair for item in p)]
                            count_n += 1

                            break

                        elif spread_value > 2:
                            pair = (pair[0], pair[1])
                            # pair = (pair[0], pair[1], pvalue, adf_result[1], kpss_result[1], spread_value)
                            invest_list.append(pair)
                            pairs = [p for p in pairs if all(item not in pair for item in p)]
                            count_n += 1

                            break

            print(len(pairs))
            print(len(invest_list))

        print(f'pvalue {count_p}')
        print(f'stationary {count_s}')
        print(f'final {count_n}')

        return invest_list


    def save_cointegrated_LS(output_dir, file, mom_data, invest_list):
        LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

        for cluster_num, firms in enumerate(invest_list):
            # Sort firms based on momentum_1
            long_short = [0] * 2
            long_short[0] = -1
            long_short[1] = 1
            # Add the data to the new table
            for i, firm in enumerate(firms):
                LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], long_short[i], cluster_num]

        firm_list_after = list(LS_table['Firm Name'])
        firm_list_before = list(mom_data.T.index)
        Missing = [item for item in firm_list_before if item not in firm_list_after]

        for i, firm in enumerate(Missing):
            LS_table.loc[len(LS_table)] = [firm, mom_data.T.loc[firm, 'Mom1'], 0, -1]

        LS_table.sort_values(by='Cluster Index', inplace=True)

        # Save the output to a CSV file in the output directory
        LS_table.to_csv(os.path.join(output_dir, file), index=False)
        print(output_dir)