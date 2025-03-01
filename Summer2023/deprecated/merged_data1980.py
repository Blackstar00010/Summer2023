
# Read ./files/merged_data.csv file and extract data after 1980-Jan-01 and save as ./files/merged_data1980.csv

import csv

# Open the input CSV file
with open('../files/price_data/merged_data.csv', 'r') as input_file:
    reader = csv.reader(input_file)

    # Create a new CSV file for writing
    with open('../files/price_data/price_data1980.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)

        # Iterate over the rows in the input file
        for i, row in enumerate(reader):
            if i < 1 or i > 1892:
                writer.writerow(row)