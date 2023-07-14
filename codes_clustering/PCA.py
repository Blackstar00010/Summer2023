import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# 1. 파일 불러오기
data = pd.read_csv('../files/momentum/2018-01.csv', header=None)
firms_list = data[data.columns[0]].tolist()[1:]
data = data.set_index(data.columns[0])
data = data[1:]
mat = data.values
mat = mat.astype(float)


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
    print('variance_ratio: ', pca.explained_variance_ratio_)
    print('sum of variance_ratio: ', np.sum(pca.explained_variance_ratio_))


# get_pd_from_pca에 넣을 columns 생성
cols = []
for i in range(1, 21):
    cols.append('pca_' + str(i))

pca_mat, pca = get_pca_data(mat, n_components=20)
pca_mat_pd = get_pd_from_pca(pca_mat, cols=cols)
pca_x = pca_mat_pd[cols]
pca = PCA(n_components=20)
pca.fit(mat)
X_pca = pca.transform(mat)

# 3. PCA 수치 검증
# TODO: for 루프로 최소 dimension 찾기
print_variance_ratio(pca)
print("original shape: ", mat.shape)
print("transformed shape: ", pca_mat.shape)

# 4. PCA 그래프 검증
mat_new = pca.inverse_transform(pca_mat)
plt.scatter(mat[:, 0], mat[:, 1], alpha=0.2)
plt.scatter(mat_new[:, 0], mat_new[:, 1], alpha=0.8)
plt.axis('equal')

# 5. StandardScaler
# 각 열의 평균과 표준편차 계산
mean = np.mean(mat, axis=0)
std = np.std(mat, axis=0)
# 데이터 표준화: (X-mu)/sigma
mat_ss = (mat - mean) / std
# PCA이후 data
mat_pd_pca = get_pd_from_pca(pca_mat)
mat_pd_pca.head(20)
mat_pd_pca_matrix = mat_pd_pca.values
# 원본 data의 첫열
first_column = data.iloc[:, 0]
first_column_matrix = np.array(first_column).reshape(-1, 1)
first_column_matrix = first_column_matrix[0:]
# 행렬 결합
combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))

# 6. PCA 결과 출력
print(mat_pd_pca)
df_combined = pd.DataFrame(combined_matrix)
df_combined.insert(0, 'Firm', firms_list)
print(df_combined)
