import pandas as pd

df = pd.read_csv('../files/history/first_day_of_month.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Loop over the dates
start_date = pd.to_datetime('1990-01-01')
end_date = pd.to_datetime('2022-12-01')
months = pd.date_range(start_date, end_date, freq='MS', tz='UTC')

for current_date in months:
    window = df.loc[:current_date].tail(48)

    mom = pd.DataFrame(index=range(1, 49), columns=window.columns)
    for i in range(1, 49):
        mom.loc[i] = window.iloc[-1] / window.shift(i).iloc[-1] - 1

    mom.index = range(1, 49)

    filename = current_date.strftime('%Y-%m') + '.csv'
    mom.to_csv('../files/momentum/' + filename, index_label='Momentum Factor')