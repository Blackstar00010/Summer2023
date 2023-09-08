import pandas as pd
import Misc.useful_stuff as us


def handle_secd(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    price_cols = ['prcod', 'prchd', 'prcld', 'prccd']
    price_shrout_cols = ['prcod', 'prchd', 'prcld', 'prccd', 'cshoc']
    bqsx_df = df[df['gvkey'] == us.ric2num('BQSX-L')]
    df = df[df['gvkey'] != us.ric2num('BQSX-L')]
    bqsx_df[price_cols] = ((bqsx_df[price_cols] < 1) * float('NaN') +
                           ((bqsx_df[price_cols] >= 1) * bqsx_df[price_cols]).replace(0, float('NaN')))

    lxkx_df = df[df['gvkey'] == us.ric2num('LXKX-L')]
    df = df[df['gvkey'] != us.ric2num('LXKX-L')]
    lxkx_df.loc[lxkx_df['cshoc'] < 10000, price_shrout_cols] = float('NaN')

    njjo_df = df[df['gvkey'] == us.ric2num('NJJO-L')]
    df = df[df['gvkey'] != us.ric2num('NJJO-L')]
    njjo_df.loc[njjo_df['prccd'] > 18, price_cols] = njjo_df.loc[njjo_df['prccd'] > 18, price_cols] * 0.060104275
    njjo_df.loc[njjo_df['prccd'] > 1844 * 0.06, price_cols] = njjo_df.loc[njjo_df['prccd'] > 1844 * 0.060104275, price_cols] * 0.010026029

    ptul_df = df[df['gvkey'] == us.ric2num('PTUL-L')]
    df = df[~df.index.isin(ptul_df.index)]
    ptul_df = ptul_df[ptul_df['datadate'] < '2022-01-01']

    lojl_df = df[df['gvkey'] == us.ric2num('LOJL-L')]
    lojl_df = lojl_df[lojl_df['datadate'] > '2007-01-01']
    df = df[~df.index.isin(lojl_df.index)]
    lojl_df.loc[lojl_df['prccd'] / lojl_df['prccd'].shift(1) > 0.2, price_shrout_cols] = float('NaN')
    lojl_df = lojl_df.dropna(subset=price_shrout_cols)

    ftlc_df = df[df['gvkey'] == us.ric2num('FTLC-L')]
    df = df[~df.index.isin(ftlc_df.index)]
    ftlc_df.loc[ftlc_df['prccd'] < 3, price_shrout_cols] = float('NaN')
    ftlc_df = ftlc_df.dropna(subset=price_shrout_cols)

    df = pd.concat([df, bqsx_df, lxkx_df, njjo_df, ptul_df, lojl_df, ftlc_df], axis=0)
    return df
