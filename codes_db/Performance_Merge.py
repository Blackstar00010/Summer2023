import pandas as pd

# Read the data from the CSV files
df1 = pd.read_csv('../files/combined_mom1_data.csv')
df2 = pd.read_csv('../files/combined_LS_data.csv')

# Save the 'Firm Name' column before dropping it
firm_names = df1['Firm Name']

# Drop the 'Firm Name' column and calculate the element-wise product
df1 = df1.drop(columns=['Firm Name'])
df2 = df2.drop(columns=['Firm Name'])

df2 = df2.shift(periods=1, axis="columns")

performance = df1.mul(df2)

# Create a new DataFrame with 'Firm Name' as the first column
df_result = pd.concat([firm_names, performance], axis=1)

# Drop columns that are all NaN
df_result = df_result.dropna(axis=1, how='all')

# Write the result to a new CSV file
df_result.to_csv('../files/combined_performance_data.csv', index=False)

