import os
import pandas as pd

# Read the data from the CSV files
df1 = pd.read_csv('../files/mom1_data_combined.csv')

directory = '../files/position_LS/equal_weight/'
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

for file in files:
    df2 = pd.read_csv(os.path.join(directory, file))

    # Find the common columns between df1 and df2
    common_columns = df1.columns.intersection(df2.columns)

    # Keep only the common columns in both dataframes
    df1 = df1[common_columns]
    df2 = df2[common_columns]

    # Shift the columns of df2 to the left
    df2 = df2.shift(periods=1, axis="columns")

    performance = df1.mul(df2)

    # Drop columns that are all NaN
    performance = performance.dropna(axis=1, how='all')

    # Write the result to a new CSV file
    performance.to_csv(os.path.join(directory, 'performance_' + file), index=False)
