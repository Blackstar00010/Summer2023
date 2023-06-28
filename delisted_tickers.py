import pandas as pd

"""
Let's write what this code does. 
If it is a scratch file, delete and make a scratch.py.
If it is a useless file, delete. 
    If it is too good to be deleted, move to ./deprecated/ folder."""

def get_data(file, start_row, end_row, column_index):
    df = pd.read_csv(file)

    data = []

    for row in range(start_row, end_row + 1):
        value = df.iloc[row - 1, column_index]
        data.append(value)

    return data


file = './files/Delisted.csv'
start_row = 1
end_row = 38501
column_index = 9

tickers = get_data(file, start_row, end_row, column_index)

file_path = "D_Tickers.csv"

df = pd.DataFrame(tickers, columns=["D_Tickers"])

df.to_csv(f"./files/D_Tickers.csv")

print("Tickers written to CSV file.")
