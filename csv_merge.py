import os
import pandas as pd

directory = './files/history'
csv_history = sorted(filename for filename in os.listdir(directory))

merged_data = pd.DataFrame()

for file in csv_history:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    if 'Date' in df.columns and 'Close' in df.columns:
        df = df[['Date', 'Close']]
        close_column_name = os.path.splitext(file)[0]
        df.rename(columns={'Close': close_column_name}, inplace=True)
        if merged_data.empty:
            merged_data = df
        else:
            merged_data = pd.merge(merged_data, df, on='Date', how='outer')

merged_data.to_csv('./files/merged_data.csv', index=False)


'''
import pandas as pd

df = pd.read_csv('./files/history/888.csv')
var = df[["Date", "Close"]]

files = './files/history/*'
'''



'''
from price_scraper import tickers

for ticker in tickers:
    files = f'./files/history/{ticker}.csv'
'''

