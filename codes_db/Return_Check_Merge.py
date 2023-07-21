import os
import pandas as pd
import matplotlib.pyplot as plt

directory = '../files/position_LS/equal_weight_performance/'
performance = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

result_df = pd.DataFrame()
file_names = []  # List to store the file names

# Loop over all files in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)

    # Calculate the average of non-NaN values in each column (excluding the 'Firm Name' column)
    column_means = df.iloc[:, 1:].mean()

    # Convert the Series of column means to a DataFrame and transpose it
    column_means_df = pd.DataFrame(column_means).T

    # Concat the means DataFrame to the result DataFrame
    result_df = pd.concat([result_df, column_means_df], ignore_index=True)

    # Remove 'performance_' and '_combined_LS' from the file name
    cleaned_filename = filename.replace('performance_', '').replace('_combined_LS', '')

    # Add the cleaned file name (without extension) to the list
    file_names.append(cleaned_filename[:-4])

# Add a new column to the result DataFrame with the file names
result_df.insert(0, 'Clustering Method', file_names)

# Separate the 'Clustering Method' column from the date columns
clustering_method = result_df['Clustering Method']
date_columns_df = result_df.drop('Clustering Method', axis=1)

# Convert the date columns to datetime format and sort them
date_columns_df.columns = pd.to_datetime(date_columns_df.columns, errors='coerce')
date_columns_df = date_columns_df.sort_index(axis=1)

# Concatenate the 'Clustering Method' column back with the sorted date columns
result_df = pd.concat([clustering_method, date_columns_df], axis=1)

# Save a new CSV file
result_df.to_csv('../files/position_Ls/result.csv', index=False)

# Plot a graph for each row
for i in range(len(result_df)):
    plt.figure(figsize=(10, 6))
    plt.plot(result_df.columns[1:], result_df.iloc[i, 1:])
    plt.title(result_df.iloc[i, 0])
    plt.xlabel('Date')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
