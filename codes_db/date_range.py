import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('../files/history/index_close_mret.csv')

# Convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set the date range
start_date = '1990-01-01'
end_date = '2022-12-01'

# Filter the data
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
filtered_df = df.loc[mask]
filtered_df = filtered_df.T
filtered_df = filtered_df.iloc[:2]

# Save the filtered data to a new CSV file
filtered_df.to_csv('../files/monthly_return_of_index.csv', index=False)

filtered_df = pd.read_csv('../files/monthly_return_of_index.csv')

# Convert the DataFrame to numeric
filtered_df = filtered_df.apply(pd.to_numeric, errors='coerce')

# Add 1 to all data values
filtered_df.iloc[:, 1:] = filtered_df.iloc[:, 1:] + 1

# Calculate the cumulative product
filtered_df.iloc[:, 1:] = filtered_df.iloc[:, 1:].cumprod(axis=1)

# Subtract 1 to get back to the original scale
filtered_df.iloc[:, 1:] = filtered_df.iloc[:, 1:] - 1

plt.plot(filtered_df.columns[1:], filtered_df.iloc[0, 1:])

plt.title('Average Values Over Time')
plt.xlabel('Date')
plt.ylabel('cumulative Value')
plt.xticks(rotation=45)
plt.show()
