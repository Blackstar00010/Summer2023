from _table_generate import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. 파일 불러오기 및 PCA함수
first=True
if first == True:
    input_dir = '../files/momentum'
    file='2018-01.csv'
    data = read_and_preprocess_data(input_dir, file)
    mat = data.values.astype(float)

    def get_pca_data(data, n_components=2):
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


    def print_variance_ratio(pca):
        print('variance_ratio: ', pca.explained_variance_ratio_)
        print('sum of variance_ratio: ', np.sum(pca.explained_variance_ratio_))


# 2. 최적 n_components 찾기
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
first_column = data.iloc[:, 0]
first_column_matrix = np.array(first_column).reshape(-1, 1)
combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
df_combined = pd.DataFrame(combined_matrix)

# Result
print(file)
print("original shape: ", mat.shape)
print("transformed shape: ", pca_mat.shape)
print_variance_ratio(pca)
print(mat_pd_pca)
print(df_combined)

# Graph after PCA
mat_new = pca.inverse_transform(pca_mat)
# Mom1-Mom2 PCA before after
plt.scatter(mat[:, 0], mat[:, 1], alpha=0.2)
plt.scatter(mat_new[:, 0], mat_new[:, 1], alpha=0.8)
plt.axis('equal')
plt.show()
