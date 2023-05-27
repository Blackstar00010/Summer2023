import yfinance as yf

# TODO
tickers = ["AZN", "HSBA", "SHEL", "OKYO"]
period = "max"

for ticker in tickers:
    equity = yf.Ticker(ticker+".L")

    # df of columns Open/High/Low/Close/Volume/Dividends/Stock Splits
    history = equity.history(period=period)
    print(history.head())
    history.to_csv('./files/history/'+ticker+".csv")
