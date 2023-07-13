import pandas as pd

# Filter out non-UK timezone values from price_data1986 file

# Read the CSV file
df = pd.read_csv('../files/history/price_data1980.csv')

# Filter rows based on Date column
df_filtered = df[df['Date'].str.contains('00:00:00\+00:00|00:00:00\+01:00')]

# Save the filtered data to a new CSV file
df_filtered.to_csv('../files/history/filtered_price_data.csv', index=False)