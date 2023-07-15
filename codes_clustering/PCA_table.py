from _table_generate import *
from sklearn.decomposition import PCA

# 파일 불러오기.
input_dir = '../files/momentum'
output_dir = '../files/Clustering/PCA(2-49)'
momentum = sorted(filename for filename in os.listdir(input_dir))

# 1. PCA 알고리즘 함수.
first = True
if first == True:
    def get_pca_data(data, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(data)
        return pca.transform(data), pca


    def get_pd_from_pca(pca_data, cols=None):
        if cols is None:
            cols = ['pca_component_{}'.format(i + 1) for i in range(pca_data.shape[1])]
        return pd.DataFrame(pca_data, columns=cols)


    def variance_ratio(pca):
        sum = np.sum(pca.explained_variance_ratio_)
        return sum

# CSV 파일 하나에 대해서 각각 실행.
for file in momentum:

    data = read_and_preprocess_data(input_dir, file)
    mat = data.values.astype(float)

    mom1 = mat[:, 0]

    # mom1을 제외한 mat/PCA(2-49)
    mat = np.delete(mat, 0, axis=1)

    # # mom49를 제외한 mat/PCA(1-48)
    # mat = np.delete(mat, 48, axis=1)

    # 2. 최적 n_components 찾기
    second = True
    if second == True:

        if len(data) < 20:
            n_components = len(data)

        else:
            n_components = 20

        while True:
            pca = PCA(n_components)
            pca.fit(mat)
            t = variance_ratio(pca)

            if t < 0.99 or n_components < 2:
                break
            else:
                n_components -= 1

        pca = PCA(n_components + 1)
        pca.fit(mat)
        t = variance_ratio(pca)
        n_components = n_components + 1

    # 3. PCA 진행 및 결과
    third = True
    if third == True:
        # get_pd_from_pca에 넣을 columns 생성
        cols = []
        for i in range(1, n_components + 1):
            cols.append('pca_' + str(i))

        pca_mat, pca = get_pca_data(mat, n_components=n_components)
        pca_mat_pd = get_pd_from_pca(pca_mat, cols=cols)
        pca_x = pca_mat_pd[cols]
        t = variance_ratio(pca)

        # Standardscaler
        # 각 열의 평균과 표준편차 계산
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        # 데이터 표준화
        mat_ss = (mat - mean) / std
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
        print(df_combined)

    # 4. PCA 결과 CSV로 저장
    forth = False
    if forth == True:
        # Columns=['Original Mom1', 'data after PCA', ...]
        output_file = os.path.join(output_dir, file)
        df_combined.to_csv(output_file, index=True)
