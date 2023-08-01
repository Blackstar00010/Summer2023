from _table_generate import *
from sklearn.cluster import DBSCAN
from dbscan_checkcheck import successful_params

dbscan = False
if dbscan:
    input_dir = '../files/PCA/PCA(1-48)'
    file = '2022-12.csv'
    data = read_and_preprocess_data(input_dir, file)
    data_array = data.values[:, 1:].astype(float)
    firm_names = data.index

    eps_values = np.linspace(0.01, 2., 20)
    min_samples_values = range(2, 20)

    successful_params = successful_params(data_array, eps_values, min_samples_values)
    print(successful_params)

    for i, example in enumerate(successful_params):
        for j in range(1):
            eps = example[j]
            ms = example[j + 1]
            print(eps)
            print(ms)


    def perform_DBSCAN2(data_array, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
        labels = dbscan.fit_predict(data_array)

        return labels, dbscan


    a, b = perform_DBSCAN2(data_array, 1.4763157894736842, 6)
    print(a)
    unique_labels = sorted(list(set(a)))

    print(len(unique_labels))

# 파일 불러오기
input_dir = '../files/PCA/PCA(1-48)_adj'
output_dir = '../files/Clustering_adj/Reversal'
Reversal = sorted(filename for filename in os.listdir(input_dir))

for file in Reversal:
    data = read_and_preprocess_data(input_dir, file)
    reversal_table_generate(data, output_dir, file)
