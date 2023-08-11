import os.path
import csv
import pandas as pd

first_day_of_month = False

if first_day_of_month:
    # Creates a new table only containing the rows of dates that are first business day of the month
    dir = "../files/history/"
    df = pd.read_csv(dir + "adj_close.csv")

    dates = df['Date']
    months = pd.DataFrame([item[5:7] for item in dates])

    flags = (months == months.shift(1)).dropna()
    df["Month Start Flag"] = flags

    df_filtered = df[df['Month Start Flag'] == False]
    df_filtered = df_filtered.drop(columns='Month Start Flag')
    df_filtered.to_csv(dir + "adj_close_first_day_of_month.csv", index=False)

momentum = True
if momentum:
    # For each month from 1990-01 to 2022-12, it creates a new table of 48 rows of momentum factor
    # Momentum Factor: ratio of the current month's value to the value from i months ago minus 1
    # mom_1 = r_{t-1}
    # mom_i = \prod_{j=t-i-1}^{t-2} (r_j+1) - 1, i \in 1,...,4

    df = pd.read_csv('../files/history/adj_close_first_day_of_month.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    start_date = pd.to_datetime('1993-01-01')
    end_date = pd.to_datetime('2022-12-01')
    months = pd.date_range(start_date, end_date)

    # jamesd: print shits = progress bar
    print('[', end='')
    for current_date in months:
        window = df.loc[:current_date].tail(50)

        mom = pd.DataFrame(index=range(1, 50), columns=window.columns)
        for i in range(1, 50):
            if i == 1:
                mom.loc[i] = window.iloc[-1] / window.iloc[-2] - 1
            else:
                mom.loc[i] = window.iloc[-2] / window.shift(i - 1).iloc[-2] - 1

        mom.index = range(1, 50)

        mom = mom.T

        # Delete rows with all NaN values
        mom = mom.dropna(how='any')

        filename = current_date.strftime('%Y-%m') + '.csv'
        mom.to_csv('../files/momentum_adj_close/' + filename, index_label='Momentum Index')

        print('-', end='')
        if int(current_date.strftime('%m')) == 12:
            print(f'] {current_date.strftime("%Y")} done!\n[')
    if int(current_date.strftime('%m')) != 12:
        print(f'] {current_date.strftime("%Y-%m")} done!\n[')

MOM_Merge = False
if MOM_Merge:
    directory = '../files/momentum_adj_close'
    long_short = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

    merged_df = pd.DataFrame()

    for file in long_short:
        data = pd.read_csv(os.path.join(directory, file))

        # Keep only the 'Momentum Index' and '1' columns
        data = data[['Momentum Index', '1']]

        file_column_name = os.path.splitext(file)[0]

        # Rename the columns
        data = data.rename(columns={'Momentum Index': 'Firm Name', '1': file_column_name})

        if merged_df.empty:
            merged_df = data
        else:
            merged_df = pd.merge(merged_df, data, on='Firm Name', how='outer')

    merged_df = merged_df.sort_values('Firm Name')

    merged_df.to_csv('../files/mom1_data_combined_adj_close.csv', index=False)

weird_value_out = False
if weird_value_out:
    directory = '../files/momentum_adj_close'
    monthly_files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

    for file in monthly_files:
        data = pd.read_csv(os.path.join(directory, file))

        comp_list = []
        for index, row in data.iterrows():
            company = row[0]
            values = [float(value) for value in row[1:]]

            if any(value > 10 for value in values[:5]):
                comp_list.append(company)

        if comp_list:
            print(file)
            '''for company in comp_list:
                print(company)'''
            print(len(comp_list))
            print()

weird_value_out_csv = False
if weird_value_out_csv:
    directory = '../files/momentum_adj_close'
    monthly_files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

    output_data = []
    for file in monthly_files:
        data = pd.read_csv(os.path.join(directory, file))

        companies_with_large_values = [] # List to store companies with values > 10

        for index, row in data.iterrows():
            company = row[0]
            values = [float(value) for value in row[1:]]

            if any(-1 < value < -0.9 for value in values[:5]):
                companies_with_large_values.append(company)

        # If there are companies that meet the criteria, add to the output data
        if companies_with_large_values:
            output_data.append([file] + companies_with_large_values)  # Combine file name with companies in one row

    # Create a DataFrame with the output data
    output_df = pd.DataFrame(output_data, columns=['Date'] + [f'{i}' for i in range(1, max(len(row) for row in output_data))])

    # Write the DataFrame to a CSV file
    output_df.to_csv('../files/SIBAL_FILES_small.csv', index=False)


# Won't be using this
return_factor = False
if return_factor:
    # For each month from 1990-01 to 2022-12, it creates a new table of 48 return factor
    # Return Factor: ratio of the current month's value to the value from 1 months ago minus 1

    df = pd.read_csv('../files/history/first_day_of_month.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    start_date = pd.to_datetime('1990-01-01')
    end_date = pd.to_datetime('2022-12-01')
    months = pd.date_range(start_date, end_date, freq='MS', tz='UTC')

    for current_date in months:
        window = df.loc[:current_date].tail(49)

        return_factor = window / window.shift() - 1
        return_factor = return_factor.iloc[::-1]
        return_factor.index = range(1, 50)

        return_factor = return_factor.T
        return_factor = return_factor.dropna(how='any')

        filename = current_date.strftime('%Y-%m') + '.csv'
        return_factor.to_csv('../files/return_factor/' + filename, index_label='Return Factor')