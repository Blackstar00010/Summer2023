import os
import pandas as pd

# Reads CSV files from './files/price_data' directory and merges them based on the 'Date' column.
# Code for sorting the date was added because the order of the date was random.

directory = '../files/history_by_ticker'
csv_history = sorted(filename for filename in os.listdir(directory))

merged_data = pd.DataFrame()

for file in csv_history:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path, encoding='latin1')
    if 'Date' in df.columns and 'Close' in df.columns:
        df = df[['Date', 'Close']]
        close_column_name = os.path.splitext(file)[0]
        df.rename(columns={'Close': close_column_name}, inplace=True)
        if merged_data.empty:
            merged_data = df
        else:
            merged_data = pd.merge(merged_data, df, on='Date', how='outer')

# Date Column Sorting (이 코드 없으면 순서가 뒤죽박죽)
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.sort_values('Date', inplace=True)
merged_data.reset_index(drop=True, inplace=True)

merged_data.to_csv('../files/price_data/merged_data.csv', index=False)
