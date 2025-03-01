
# Reads data from CSV file of delisted companies and converts into an SQL file.

import pandas as pd


def csv_to_sql(csv_file, sql_file):
    df = pd.read_csv(csv_file)
    sql_ = []
    for index, row in df.iterrows():
        values = ", ".join(f"'{str(value)}'" for value in row.values)
        sql_.append(values)

    with open(sql_file, 'w') as file:
        file.write("\n".join(sql_))


'''
csv_file = './files/merged_data.csv'
sql_file = './files/merged_data.sql'
'''
csv_file = '../files/Delisted.csv'
sql_file = '../files/Delisted.sql'

csv_to_sql(csv_file, sql_file)
