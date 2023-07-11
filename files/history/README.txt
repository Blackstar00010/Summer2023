IF YOU WANT TO FETCH FRESHLY UPDATED DATE-FIRMS PRICE MATRIX, USE:

import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

asdf = pd.read_csv('https://www.jamesd002.com/file/price_data1983.csv')
print(asdf.head())


OR, IF YOU WANT TO DOWNLOAD THE CSV FILE, VISIT:

https://www.jamesd002.com/file/download.html