import logging
from prep import Prep
import yfinance as yf

import openpyxl
import pandas as pd
import os

def get_data (file, sheet, start_row, end_row, column_index):
    df = pd.read_excel(file, sheet_name=sheet)

    data = []

    for row in range(start_row, end_row+1):
        value = df.iloc[row-1,column_index]
        value = value.rstrip(" (LSE)") # 뒤에 달린 (LSE) 삭제
        data.append(value)

    return data

#df1 = pd.read_excel('/Users/yoonsanglee/Documents/GitHub/Summer2023/files/SPGlobal_ListManager-All_27-May-2023.xlsx')
file = '/Users/yoonsanglee/Documents/GitHub/Summer2023/files/SPGlobal_ListManager-All_27-May-2023.xlsx'
sheet = 'List Manager - Companies'
start_row = 8
end_row = 662
column_index = 1

tickers = get_data(file, sheet, start_row, end_row, column_index)

#print(tickers)


# TODO : extract all tickers from the .xlsx file
# (lse) 빼고 찾기
# tickers = ["AZN", "HSBA", "SHEL", "OKYO"]
# tickers = ["ADMR", "AAF"]
period = "max"

for ticker in tickers:
    equity = yf.Ticker(ticker+".L")

    # df of columns Open/High/Low/Close/Volume/Dividends/Stock Splits
    history = equity.history(period=period, end="2022-12-31")
    history = history.round(3)
    logging.info(f"{ticker} data completed")
    history.to_csv(f"./files/history/{ticker}.csv")
