import pandas as pd

# Read the CSV file
df = pd.read_csv('./files/history/price_data1986.csv')

# Filter rows based on Date column
df_filtered = df[df['Date'].str.contains('00:00:00\+00:00|00:00:00\+01:00')]

# Save the filtered data to a new CSV file
df_filtered.to_csv('./files/history/filtered_price_data.csv', index=False)

"""
import pandas as pd

dir = "./files/"

df = pd.read_csv(dir+"filtered_price_data.csv")
dates = df['Date']
months = pd.DataFrame([item[5:7] for item in dates])
flags = (months == months.shift(1)).dropna()
df["Month Start Flag"] = flags
df.to_csv(dir+"asdf.csv")
"""




