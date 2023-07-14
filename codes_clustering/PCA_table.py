from _table_generate import *
from _PCA import *
# 1. 파일 불러오기
input_dir = '../files/momentum'
output_dir = '../files/Clustering/PCA'
momentum = sorted(filename for filename in os.listdir(input_dir))

# 2. CSV 파일 하나에 대해서 각각 실행
for file in momentum:
    data = read_and_preprocess_data(input_dir, file)
    mat = data.values

    mc = True
    if mc == True:
        # 4. PCA 알고리즘 구현

        while True:
            if len(data)<20:
                n_components=len(data)
                pca_mat, pca = get_pca_data(mat, n_components=n_components)
                t = print_variance_ratio(pca)

                if t < 0.99:
                    break

            




            else:
                n_components=20

        # get_pd_from_pca에 넣을 columns 생성
        cols = []
        for i in range(1, n_components+1):
            cols.append('pca_' + str(i))

    md = True
    if md == True:
        pca_mat, pca = get_pca_data(mat, n_components=n_components)
        pca_mat_pd = get_pd_from_pca(pca_mat, cols=cols)
        pca_x = pca_mat_pd[cols]

        t= print_variance_ratio(pca)
        print(t)
        print(n_components)



    mk=False
    if mk == True:

        print("original shape: ", mat.shape)
        print("transformed shape: ", pca_mat.shape)

        # 5. Standardscaler
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

    third=False
    if third==True:
        # 6. Result CSV 생성
        # Columns=['Original Mom1', 'data after PCA', ...]
        output_file = os.path.join(output_dir, file)
        df_combined = pd.DataFrame(combined_matrix)


        df_combined.to_csv(output_file, index=True)
