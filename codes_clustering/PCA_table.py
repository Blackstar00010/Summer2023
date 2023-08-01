from _table_generate import *
from sklearn.decomposition import PCA
from PCA_single import *

# 파일 불러오기
input_dir = '../files/momentum_adj'
output_dir = '../files/PCA/PCA(1-48)_adj'
momentum = sorted(filename for filename in os.listdir(input_dir))

# CSV 파일 하나에 대해서 각각 실행.
for file in momentum:
    if file == '.DS_Store':
        continue

    data = read_and_preprocess_data(input_dir, file)

    mom1 = data.values.astype(float)[:, 0]
    data_normalized = (data - data.mean()) / data.std()

    mat = data_normalized.values.astype(float)

    # mom1을 제외한 mat/PCA(2-49)
    # mat = np.delete(mat, 0, axis=1)

    # # mom49를 제외한 mat/PCA(1-48)
    mat = np.delete(mat, 48, axis=1)

    # 1. Searching optimal n_components
    if len(data) < 20:
        n_components = len(data)

    else:
        n_components = 20

    pca = PCA(n_components)
    pca.fit(mat)
    t = variance_ratio(pca)

    while t > 0.99:
        n_components -= 1
        pca = PCA(n_components)
        pca.fit(mat)
        t = variance_ratio(pca)

    while t < 0.99:
        n_components += 1
        pca = PCA(n_components)
        pca.fit(mat)
        t = variance_ratio(pca)

    # 2. PCA
    # get_pd_from_pca에 넣을 columns 생성
    cols = []
    for i in range(1, n_components + 1):
        cols.append('pca_' + str(i))

    pca_mat, pca = get_pca_data(mat, n_components=n_components)
    pca_mat_pd = get_pd_from_pca(pca_mat, cols=cols)

    # PCA이후 data
    mat_pd_pca = get_pd_from_pca(pca_mat)
    mat_pd_pca.head()
    mat_pd_pca_matrix = mat_pd_pca.values

    # Original Mom1 Combining
    first_column = mom1
    first_column_matrix = np.array(first_column).reshape(-1, 1)
    combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
    df_combined = pd.DataFrame(combined_matrix)
    df_combined.index = data.index

    print(file)
    print(t)
    print(n_components)

    # 4. Save CSV
    # Column format: ['Original Mom1', 'data after PCA', ...]
    output_file = os.path.join(output_dir, file)
    df_combined.to_csv(output_file, index=True)
