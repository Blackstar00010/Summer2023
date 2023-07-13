import logging
import yfinance as yf
import pandas as pd

# Fetches tickers from the .xlsx file downloaded from CapitalIQ, finds ISINs and save as a .csv file.
# Then, for each ticker, it fetches price data from yahoo finance and save as a .csv file.


def fetch_ticker(file: str, sheet: str, start_row: int, end_row: int, column_index: int):
    """
    Fetches tickers from the .xlsx file downloaded from CapitalIQ.
    :param file: string of filename with relative directory
    :param sheet: string of the name of the sheet
    :param start_row: the first row to extract ticker
    :param end_row: the last row to extract ticker
    :param column_index: the column that contains ticker information
    :return: list of tickers
    """
    df = pd.read_excel(file, sheet_name=sheet)

    data = []

    for row in range(start_row, end_row + 1):
        value = df.iloc[row - 1, column_index]
        value = value.replace(" (LSE)", "").rstrip(".").replace(".", "-")
        data.append(value)

    return data


file = '../files/tickers_from_ciq.xlsx'
sheet = 'List Manager - Companies'
start_row = 8
end_row = 662
column_index = 1

tickers = fetch_ticker(file, sheet, start_row, end_row, column_index)
isins = []
for i, ticker in enumerate(tickers):
    if i % 10 == 9:
        print(f"{i + 1}/{len(tickers)} completed")
    try:
        isins.append(yf.Ticker(ticker + ".L").isin)
    except:
        isins.append("NULL")
file_name = "0_tickers.csv"

df = pd.DataFrame(tickers, columns=["Ticker"])
df["ISIN"] = isins

df.to_csv(f"../files/history/" + file_name)

print("Tickers written to CSV file.")

period = "max"

for ticker in tickers:
    equity = yf.Ticker(ticker + ".L")

    history = equity.history(period=period, end="2023-06-30")
    history = history.round(3)
    logging.info(f"{ticker} data completed")
    history.to_csv(f"../files/history/{ticker}.csv")
