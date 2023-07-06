import pandas as pd

# Read the CSV file
df = pd.read_csv('./files/price_data1986.csv')

# Filter rows based on Date column
df_filtered = df[df['Date'].str.contains('00:00:00\+00:00|00:00:00\+01:00')]

# Save the filtered data to a new CSV file
df_filtered.to_csv('./files/filtered_price_data.csv', index=False)






