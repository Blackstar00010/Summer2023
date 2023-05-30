import logging
from prep import Prep
sys_setup = Prep(should_prep=True)
import yfinance as yf


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

sys_setup.revert_path()
