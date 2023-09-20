import os.path
import csv
import pandas as pd


weird_value_out = False
if weird_value_out:
    directory = '../files/characteristics'
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
    directory = '../files/characteristics'
    monthly_files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

    output_data = []
    for file in monthly_files:
        data = pd.read_csv(os.path.join(directory, file))

        companies_with_large_values = [] # List to store companies with values > 10

        for index, row in data.iterrows():
            company = row[0]
            values = [float(value) for value in row[1:]]

            if any(value > 10 for value in values[:5]):
                companies_with_large_values.append(company)

        # If there are companies that meet the criteria, add to the output data
        if companies_with_large_values:
            output_data.append([file] + companies_with_large_values)  # Combine file name with companies in one row

    # Create a DataFrame with the output data
    output_df = pd.DataFrame(output_data, columns=['Date'] + [f'{i}' for i in range(1, max(len(row) for row in output_data))])

    # Write the DataFrame to a CSV file
    output_df.to_csv('../files/SIBAL_FILES_small.csv', index=False)

return_factor = False
if return_factor:
    # For each month from 1990-01 to 2022-12, it creates a new table of 48 return factor
    # Return Factor: ratio of the current month's value to the value from 1 months ago minus 1

    df = pd.read_csv('../files/price_data/first_day_of_month.csv')
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