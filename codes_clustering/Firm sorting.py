#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import glob
import datetime
import numpy as np
import urllib.parse
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


csv_directory = 'C:/Users/김주환/Desktop/My files/raw_data'
#csv_directory = 'C:/Users/IE/Desktop/My files/raw_data'
csv_data_dict = {}


for file in os.listdir(csv_directory):
    if file.endswith('.csv'):
        file_path = os.path.join(csv_directory, file)
        df = pd.read_csv(file_path)
        csv_data_dict[file.replace('.csv', '')] = df

sorted_csv_data_dict = dict(sorted(csv_data_dict.items(), key=lambda x: x[0]))

for key, value in sorted_csv_data_dict.items():
    print(f"CSV 파일: {key}")
    print(value)
    print()


# In[3]:


csv_names=list(sorted_csv_data_dict.keys())
print(len(csv_names))
print(csv_names)


# In[4]:


values=list(sorted_csv_data_dict.values())
print(values[1])


# In[5]:


my_dict = {}
for index, row in values[1].iterrows():
    if row.isnull().any():
        empty_row = row[row.isnull()].index.tolist()
        my_dict['firm'+str(index+1)] = empty_row

print(my_dict)


# In[6]:


# my_dict = {}
# for index, row in values[0].iterrows():
#     zero_columns = row[row == 0].index.tolist()
#     if zero_columns:
#         zero_columns = row[row == 0].index.tolist()
#         my_dict['firm' + str(index + 1)] = zero_columns

# print(my_dict)


# In[7]:


total = []

for i in range(len(csv_names)):
    total.append({})

print(total)

for i in range(len(csv_names)):
    for index, row in values[i].iterrows():
        if row.isnull().any():
            empty_row = row[row.isnull()].index.tolist()
            my_dict['firm'+str(index+1)] = empty_row

print(total)


# In[8]:


result_dict = {key: value for key, value in zip(sorted_csv_data_dict.keys(), total)}
print(result_dict)


# In[9]:


output_file = 'combined_data.csv'

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'Firm', 'Characteristics'])
    for filename, data in result_dict.items():
        for firm, characteristics in data.items():
            writer.writerow([filename, firm] + characteristics)

print(f"CSV file '{output_file}' has been created.")

# Generate download link
download_link = urllib.parse.quote(output_file)
file_path = os.path.abspath(output_file)
print(f"Download link: {download_link}")
print(f"File path: {file_path}")

