import os
import pandas as pd
import numpy as np


def read_and_preprocess_data(input_dir, file):
    # Read data from CSV file
    data = pd.read_csv(os.path.join(input_dir, file), index_col=0)

    # Replace infinities with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    data.dropna(inplace=True)

    return data


# Generates new table with firm list / momentum_1 / Long, Short Index / Cluster Index
def new_table_generate(data, clusters, output_dir, file):
    # New table with firm name, mom_1, long and short index, cluster index
    LS_table = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index'])

    for cluster_num, firms in enumerate(clusters):
        # Sort firms based on momentum_1
        firms_sorted = sorted(firms, key=lambda x: data.loc[x, '1'])
        long_short = [0] * len(firms_sorted)
        for i in range(len(firms_sorted) // 2):
            long_short[i] = 1  # -1 to the high ones
            long_short[-i - 1] = -1  # 1 to the low ones
            # 0 to middle point when there are odd numbers in a cluster

        # Add the data to the new table
        for i, firm in enumerate(firms_sorted):
            LS_table.loc[len(LS_table)] = [firm, data.loc[firm, '1'], long_short[i], cluster_num + 1]

    # Save the output to a CSV file in the output directory
    LS_table.to_csv(os.path.join(output_dir, file), index=False)


def reversal_table_generate(data, output_dir, file):
    LS_table_reversal = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short'])
    firm_lists = data.index
    firm_sorted = sorted(firm_lists, key=lambda x: data.loc[x, '0'])
    long_short = [0] * len(firm_sorted)
    t=int(len(firm_lists)*0.1)
    for i in range(t):
        long_short[i] = 1
        long_short[-i - 1] = -1

    for i, firm in enumerate(firm_sorted):
        LS_table_reversal.loc[len(LS_table_reversal)] = [firm, data.loc[firm, '0'], long_short[i]]

    # Save the output to a CSV file in the output directory
    LS_table_reversal.to_csv(os.path.join(output_dir, file), index=False)
