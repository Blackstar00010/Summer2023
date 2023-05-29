import yfinance as yf
import logging
from prep import Prep


logging.basicConfig(level=logging.INFO)

_ = Prep()
_.check_openssl()
_.update_path()
_.check_path()
_.check_openssl()

# TODO : extract all tickers from the .xlsx file
tickers = ["AZN", "HSBA", "SHEL", "OKYO"]
period = "max"

for ticker in tickers:
    equity = yf.Ticker(ticker+".L")

    # df of columns Open/High/Low/Close/Volume/Dividends/Stock Splits
    history = equity.history(period=period, end="2022-12-31")
    history = history.round(3)
    logging.info(f"{ticker} data completed")
    history.to_csv(f"./files/history/{ticker}.csv")

_.revert_path()
