import yfinance as yf
import logging


logging.basicConfig(level=logging.INFO)

# TODO : extract all tickers from the .xlsx file
tickers = ["AZN", "HSBA", "SHEL", "OKYO"]
period = "max"

for ticker in tickers:
    equity = yf.Ticker(ticker+".L")

    # df of columns Open/High/Low/Close/Volume/Dividends/Stock Splits
    history = equity.history(period=period)
    logging.info(f"{ticker} data completed")
    history.to_csv(f"./files/history/{ticker}.csv")
