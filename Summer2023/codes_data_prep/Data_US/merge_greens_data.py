import pandas as pd
import useful_stuff as us
from pathlib import Path
import platform


def collapse_dup(some_df: pd.DataFrame, subset: list = None, df_name: str = None) -> pd.DataFrame:
    """
    Use parallel computing to groupby based on subset and collapse duplicates
    :param some_df: dataframe with duplicated rows
    :param subset: subset of columns to consider duplicates
    :param df_name: name of dataframe. If not None, prints progress
    :return: dataframe with duplicates collapsed
    """
    if subset is None:
        subset = some_df.columns.tolist()
    some_df = some_df.sort_values(by=subset).reset_index(drop=True)

    # separating dirty and clean parts so that we only have to compute the dirty part
    dirty_index = some_df.index[some_df.duplicated(subset, keep=False)]
    dirty_part = some_df.loc[dirty_index, :]
    ret = some_df.drop(dirty_index, axis=0)

    # dropping all-NA rows
    dirty_part = dirty_part.dropna(subset=[col for col in dirty_part.columns if col not in subset], how='all')
    dirty_part = dirty_part.sort_values(by=subset).reset_index(drop=True)

    # ffill-ing each groups of duplicates
    dirty_part['group_no'] = dirty_part.groupby(subset).ngroup()
    dirty_part = dirty_part.groupby('group_no').apply(lambda group: group.fillna(method='ffill')).reset_index(drop=True)
    dirty_part = dirty_part.drop_duplicates(subset=subset, keep='last').drop('group_no', axis=1)

    ret = pd.concat([ret, dirty_part], axis=0).sort_values(by=subset).reset_index(drop=True)
    if df_name is not None:
        print(f'Finished managing duplicates of {df_name} data!')
    return ret


if __name__ == '__main__':
    convert_to_csv = True
    if convert_to_csv:
        files_dir = '/Users/jamesd/Desktop/wrds_data/us/'
        data1 = pd.read_csv(files_dir + 'data.csv', low_memory=False)
        data2 = pd.read_csv(files_dir + 'data2.csv', low_memory=False)
        data3 = pd.read_csv(files_dir + 'data3.csv', low_memory=False)
        data4 = pd.read_csv(files_dir + 'data4.csv', low_memory=False)
        data5 = pd.read_csv(files_dir + 'data5.csv', low_memory=False)
        data6 = pd.read_csv(files_dir + 'data6.csv', low_memory=False)

        # join all data
        target_cols = ['permno', 'gvkey', 'datadate', 'absacc', 'acc', 'aeavol', 'age', 'agr', 'baspread', 'beta',
                       'betasq', 'bm', 'bm_ia', 'cash', 'cashdebt', 'cashpr', 'cfp', 'cfp_ia', 'chatoia', 'chcsho',
                       'chempia', 'chinv', 'chmom', 'chpmia', 'chtx', 'cinvest', 'convind', 'currat', 'depr', 'divi',
                       'divo', 'dolvol', 'dy', 'ear', 'egr', 'ep', 'gma', 'herf', 'hire', 'idiovol', 'ill', 'indmom',
                       'invest', 'IPO', 'lev', 'lgr', 'maxret', 'ms', 'mve', 'mve_ia', 'nincr', 'operprof',
                       'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale', 'pchquick', 'pchsale_pchrect', 'pctacc',
                       'pricedelay', 'ps', 'quick', 'rd', 'retvol', 'roaq', 'roeq', 'roic', 'rsup', 'salecash', 'salerec', 'secured', 'securedind', 'sgr',
                       'sin', 'SP', 'std_dolvol', 'std_turn', 'sue', 'tang', 'tb', 'turn', 'zerotrade']
        data_list = []
        for data in [data1, data2, data3, data4, data5, data6]:
            specific_col = [col for col in data.columns if col in target_cols]
            if specific_col != ['gvkey', 'datadate']:
                data_list.append(data[specific_col])
        all_df = pd.concat(data_list, axis=0)

        # permno to join with close
        permno_df = all_df[['permno', 'gvkey']].drop_duplicates(subset=['permno', 'gvkey'], keep='first')
        permno_df = permno_df[permno_df['permno'].notna()]
        # print(permno_df[permno_df.duplicated(subset=['permno'], keep=False)])
        permno_df = permno_df.drop_duplicates(subset=['permno'], keep='last')
        all_df = all_df[all_df['permno'].isin(permno_df['permno'])]

        # match format & collapse duplicates
        all_df = us.change_date_format(all_df, df_name='all_df', date_col_name='datadate')
        all_df['datadate'] = all_df['datadate'].str[:7]
        all_df = collapse_dup(all_df, subset=['gvkey', 'datadate'], df_name='all_df')

        close_df = pd.read_csv(files_dir + 'close.csv', low_memory=False)
        close_df = close_df.rename(columns={'PERMNO': 'permno', 'PRC': 'prc', 'DATE': 'datadate'})
        close_df = us.change_date_format(close_df, df_name='close_df', date_col_name='datadate')
        close_df = close_df[close_df['permno'].isin(permno_df['permno'])]
        close_df = close_df[close_df['prc'] > 0]
        close_df = close_df[close_df['datadate'].str[5:7] != close_df['datadate'].str[5:7].shift(-1)]
        close_df['datadate'] = close_df['datadate'].str[:7]
        # print(close_df)
        for i in range(1, 49):
            close_df[f'mom{i}'] = close_df.groupby('permno')['prc'].shift(i)
            close_df[f'mom{i}'] = close_df['prc'] / close_df[f'mom{i}'] - 1
        close_df = close_df.drop('prc', axis=1)

        export_dir = str(Path(__file__).parents[2]) + '/files/characteristics_us/' if platform.system() == 'Darwin' else str(Path(__file__).parents[2]) + '\\files\\characteristics_us\\'
        for adate in all_df['datadate'].unique():
            this_month_df = all_df[all_df['datadate'] == adate].drop('datadate', axis=1)
            this_month_price_df = close_df[close_df['datadate'] == adate].drop('datadate', axis=1)
            this_month_df = this_month_df.merge(this_month_price_df, on='permno', how='left')
            this_month_df = this_month_df.drop('gvkey', axis=1)
            this_month_df = this_month_df.drop_duplicates(subset=['permno'], keep='last')
            this_month_df = this_month_df.rename(columns={'permno': 'firms'})
            this_month_df = this_month_df.set_index('firms')
            this_month_df.to_csv(export_dir + f'/{adate}.csv', index=True, index_label='firms')

    clean_data = True
    if clean_data:
        characteristics_dir = str(
            Path(__file__).parents[2]) + '/files/characteristics_us/' if platform.system() == 'Darwin' else str(
            Path(__file__).parents[2]) + '\\files\\characteristics_us\\'
        first_days = us.listdir(characteristics_dir)
        for afirst_day in first_days:
            # importing momentum data
            chars_df = pd.read_csv(characteristics_dir + afirst_day, low_memory=False)
            chars_df = chars_df.set_index('firms')

            best_case_finder = []
            chars_df_og = chars_df
            for thresh_col in range(10, 1, -1):
                chars_df = chars_df_og
                # drop columns(characteristics) with too many NaNs
                for acol in chars_df.columns:
                    # shitty column based on thresh_col
                    if chars_df[acol].notna().sum() < (len(chars_df) * thresh_col / 10):
                        chars_df = chars_df.drop(acol, axis=1)

                # drop rows(firms) with too many NaNs
                thresh_row = 1.0
                temp_df = chars_df.dropna(thresh=int(chars_df.shape[1] * thresh_row), axis=0)
                while len(temp_df) < (len(chars_df) / 2):
                    thresh_row -= 0.1
                    temp_df = chars_df.dropna(thresh=int(chars_df.shape[1] * thresh_row), axis=0)
                temp_df = temp_df.dropna(how='any', axis=1)

                best_case_finder.append([thresh_col, thresh_row, temp_df.shape[0] * temp_df.shape[1]])
            best_case_finder = pd.DataFrame(best_case_finder, columns=['thresh_col', 'thresh_row', 'valid_count'])
            best_case_finder = best_case_finder.sort_values(by=['valid_count', 'thresh_col'], ascending=False)

            chars_df = chars_df_og
            for acol in chars_df.columns:
                if chars_df[acol].notna().sum() < (len(chars_df) * best_case_finder.iloc[0, 0] / 10):
                    chars_df = chars_df.drop(acol, axis=1)
            chars_df = chars_df.dropna(thresh=int(chars_df.shape[1] * best_case_finder.iloc[0, 1]), axis=0)
            chars_df = chars_df.dropna(how='any', axis=1)
            chars_df.to_csv(characteristics_dir + afirst_day, index=True, index_label='firms')
            # print(f'{afirst_day} exported!')