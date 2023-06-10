import logging
from prep import Prep
import yfinance as yf

import openpyxl
# TODO:뭔갈 임포트 했으면 requirements.txt 수정해야지
import pandas as pd
import os

# 엑셀 종목들을 tickers라는 list로
def get_data (file, sheet, start_row, end_row, column_index):#TODO:PEP수정
    df = pd.read_excel(file, sheet_name=sheet)

    data = []

    for row in range(start_row, end_row+1):
        value = df.iloc[row-1,column_index]#TODO:PEP수정
        value = value.replace(" (LSE)", "").rstrip(".").replace(".","-")#TODO:PEP수정
        data.append(value)

    return data
#TODO:PEP수정
#df1 = pd.read_excel('/Users/yoonsanglee/Documents/GitHub/Summer2023/files/SPGlobal_ListManager-All_27-May-2023.xlsx')
#TODO:누군가에게는 Uers/yoonsanglee 라는 폴더가 없다네요
file = '/Users/yoonsanglee/Documents/GitHub/Summer2023/files/SPGlobal_ListManager-All_27-May-2023.xlsx'
sheet = 'List Manager - Companies'
start_row = 8
end_row = 662
column_index = 1

tickers = get_data(file, sheet, start_row, end_row, column_index)

#print(tickers)

# tickers list를 csv로
file_path = "tickers.csv"

df = pd.DataFrame(tickers, columns=["Ticker"])

df.to_csv(f"./files/history/*tickers.csv")

print("Tickers written to CSV file.")



# TODO : extract all tickers from the .xlsx file, 투 매니 블랭크 라인이라고 나는 뜨는데
# (lse) 빼고 찾기
# tickers = ["AZN", "HSBA", "SHEL", "OKYO"]
# tickers = ["ADMR", "AAF"]
period = "max"

for ticker in tickers:
    equity = yf.Ticker(ticker+".L")

    # df of columns Open/High/Low/Close/Volume/Dividends/Stock Splits
    history = equity.history(period=period, end="2023-06-20")
    history = history.round(3)
    logging.info(f"{ticker} data completed")
    history.to_csv(f"./files/history/{ticker}.csv")
