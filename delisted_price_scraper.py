import requests
import json
import csv
# import logging
# import FinanceDataReader as fdr
# import yfinance as yf
import pandas as pd



def get_data(file, start_row, end_row, column_index):
    df = pd.read_csv(file)

    data = []

    for row in range(start_row, end_row + 1):
        value = df.iloc[row - 1, column_index]
        data.append(value)

    return data


file = './files/Delisted.csv'
start_row = 1
end_row = 38501
column_index = 9

tickers = get_data(file, start_row, end_row, column_index)


# 어디선가 베낀 코드 (작동 안 함, 출처: 기억 안 남)
api_key = '7f1c7d8d7b34153637d45062b06151c3:6390c950b3f6456c2ee9c9a3315a81c3'

for ticker in tickers:
    url = f'https://api.gurufocus.com/public/user/{api_key}/{ticker}/price'
    response = requests.get(url)
    data = response.json()['data']

    if 'error' in data:
        print(f"Error retrieving data for {ticker}: {data['error']}")
        continue

    header = ['date', 'open', 'close']
    rows = [[item['date'], item['open'], item['close']] for item in data]

    with open(f"{ticker}_prices.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"{ticker} data completed")

'''
period = "max"

for ticker in tickers:
    equity = yf.Ticker(ticker + ".L")

    history = equity.history(period=period, end="2023-06-20")
    history = history.round(3)
    logging.info(f"{ticker} data completed")
    history.to_csv(f"./files/Delisted/{ticker}.csv")

'''

'''
period = "max"

for ticker in tickers:
    ticker_with_exchange = ticker + ".L"

    history = fdr.DataReader(ticker_with_exchange, start=None, end="2023-06-20")
    history = history.round(3)
    logging.info(f"{ticker} data completed")
    history.to_csv(f"./files/Delisted/{ticker}.csv")
'''

