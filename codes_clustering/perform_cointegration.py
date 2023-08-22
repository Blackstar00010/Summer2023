import os
from PCA_and_ETC import *

# input_dir = '../files/momentum_adj'
# output_dir = '../files/Clustering_adj/Cointegration'
input_dir = '../files/momentum_adj_close'
output_dir = '../files/Clustering_adj_close/Cointegration'

files = sorted(filename for filename in os.listdir(input_dir))
is_jamesd = 'jamesd' in os.path.abspath('.')
for file in files:
    print(file)
    if file in os.listdir(output_dir):
        continue
    data = read_and_preprocess_data(input_dir, file)

    mom_data = read_mom_data(data)

    # inv_list = find_cointegrated_pairs_deprecated(mom_data)
    inv_list = find_cointegrated_pairs(mom_data)

    LS_Table = True
    if LS_Table:
        save_cointegrated_LS(output_dir, file, mom_data, inv_list)

