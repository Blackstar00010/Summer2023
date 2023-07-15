from _table_generate import *
from sklearn.decomposition import PCA

# 1. 파일 불러오기
input_dir = '../files/momentum'
output_dir = '../files/Clustering/PCA'
momentum = sorted(filename for filename in os.listdir(input_dir))

# 2. PCA 알고리즘 구현
def get_pca_data(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.transform(data), pca


def get_pd_from_pca(pca_data, cols=None):
    if cols is None:
        cols = ['pca_component_{}'.format(i + 1) for i in range(pca_data.shape[1])]
    return pd.DataFrame(pca_data, columns=cols)


def print_variance_ratio(pca):
    sum=np.sum(pca.explained_variance_ratio_)
    # 'variance_ratio: ', pca.explained_variance_ratio_
    # 'sum of variance_ratio: ', np.sum(pca.explained_variance_ratio_)
    return sum

# 3. CSV 파일 하나에 대해서 각각 실행
for file in momentum:
    data = read_and_preprocess_data(input_dir, file)
    mat = data.values

    # 4. 최적 n_components 찾기
    mc = True
    if mc == True:

        if len(data)<20:
            n_components = len(data)

        else:
            n_components = 20

        while True:
            pca = PCA(n_components)
            pca.fit(mat)
            t = print_variance_ratio(pca)

            if t < 0.99 or n_components < 2:
                break
            else:
                n_components-=1

        pca = PCA(n_components+1)
        pca.fit(mat)
        t = print_variance_ratio(pca)
        n_components=n_components+1

        # print(t)
        # print(n_components)

    # 5. PCA 진행
    md = True
    if md == True:
        # get_pd_from_pca에 넣을 columns 생성
        cols = []
        for i in range(1, n_components+1):
            cols.append('pca_' + str(i))

        pca_mat, pca = get_pca_data(mat, n_components=n_components)
        pca_mat_pd = get_pd_from_pca(pca_mat, cols=cols)
        pca_x = pca_mat_pd[cols]
        t= print_variance_ratio(pca)

        # print(t)
        # print(n_components)

    # 6. Standardscaler 및 결과 확인
    mk=True
    if mk == True:

        print("original shape: ", mat.shape)
        print("transformed shape: ", pca_mat.shape)

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
        # 원본 data의 첫열
        first_column = data.iloc[:, 0]
        first_column_matrix = np.array(first_column).reshape(-1, 1)
        first_column_matrix = first_column_matrix[0:]
        # 행렬 결합
        combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))

        print(file)
        print(t)
        print(mat_pd_pca)
        df_combined = pd.DataFrame(combined_matrix)
        print(df_combined)

    #결과 출력
    third=True
    if third==True:
        # 6. Result CSV 생성
        # Columns=['Original Mom1', 'data after PCA', ...]
        output_file = os.path.join(output_dir, file)
        df_combined = pd.DataFrame(combined_matrix)
        df_combined.to_csv(output_file, index=True)
