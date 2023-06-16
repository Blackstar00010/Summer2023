import logging
import yfinance as yf
import pandas as pd

# pip freeze > requirements. txt

def get_data(file, sheet, start_row, end_row, column_index):
    df = pd.read_excel(file, sheet_name=sheet)

    data = []

    for row in range(start_row, end_row + 1):
        value = df.iloc[row - 1, column_index]
        value = value.replace(" (LSE)", "").rstrip(".").replace(".", "-")
        data.append(value)

    return data


file = './files/SPGlobal_ListManager-All_27-May-2023.xlsx'
sheet = 'List Manager - Companies'
start_row = 8
end_row = 662
column_index = 1

tickers = get_data(file, sheet, start_row, end_row, column_index)

file_path = "tickers.csv"

df = pd.DataFrame(tickers, columns=["Ticker"])

df.to_csv(f"./files/history/_tickers.csv")

print("Tickers written to CSV file.")

period = "max"

for ticker in tickers:
    equity = yf.Ticker(ticker + ".L")

    history = equity.history(period=period, end="2023-06-20")
    history = history.round(3)
    logging.info(f"{ticker} data completed")
    history.to_csv(f"./files/history/{ticker}.csv")
