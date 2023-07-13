#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import csv
import urllib.parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# 파일 불러오기
#raw_data_dir = 'C:/Users/김주환/Desktop/My files/raw_data'
#pca_output_dir = 'C:/Users/김주환/Desktop/My files/PCA'
raw_data_dir = 'C:/Users/IE/Desktop/My files/raw_data'
pca_output_dir = 'C:/Users/IE/Desktop/My files/PCA'
csv_files = [file for file in os.listdir(raw_data_dir) if file.endswith('.csv')]


# CSV 파일 하나에 대해서 각각 실행
for file in csv_files: 

    # CSV파일 df변환
    csv_path = os.path.join(raw_data_dir, file)
    data = pd.read_csv(csv_path, header=None)
    # 회사이름 추출 후 Value만 가지고 있는 dataframe생성.
    firms_list = data[data.columns[0]].tolist()[1:]
    data = data.set_index(data.columns[0])
    data = data[1:]
    # df 행렬변환
    mat = data.values.astype(float)
    
    
    # PCA 알고리즘 구현
    def get_pca_data(data, n_components=2):
        pca = PCA(n_components=n_components)
        pca.fit(data)
        return pca.transform(data), pca


    def get_pd_from_pca(pca_data, cols=None):
        if cols is None:
            cols = ['pca_component_{}'.format(i+1) for i in range(pca_data.shape[1])]
        return pd.DataFrame(pca_data, columns=cols)


    def print_variance_ratio(pca):
        print('variance_ratio: ', pca.explained_variance_ratio_)
        print('sum of variance_ratio: ', np.sum(pca.explained_variance_ratio_))


    cols = []
    for i in range(1, 21):
        cols.append('pca_' + str(i))  
    pca_mat, pca= get_pca_data(mat, n_components=20)
    pca_mat_pd = get_pd_from_pca(pca_mat, cols=cols)
    pca_x= pca_mat_pd[cols]
    print_variance_ratio(pca)
    print("original shape: ", mat.shape)
    print("transformed shape: ", pca_mat.shape)

    
    # Standardscaler
    # 각 열의 평균과 표준편차 계산
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0)
    # 데이터 표준화
    mat_ss = (mat - mean) / std
    #PCA이후 data
    mat_pd_pca = get_pd_from_pca(pca_mat)
    mat_pd_pca.head(20)
    mat_pd_pca_matrix = mat_pd_pca.values
    #원본 data의 첫열
    first_column = data.iloc[:, 0]
    first_column_matrix= np.array(first_column).reshape(-1, 1)
    first_column_matrix=first_column_matrix[0:]
    # 행렬 결합
    combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
    
    
    # Result CSV 생성
    output_file = os.path.join(pca_output_dir, file)
    df_combined = pd.DataFrame(combined_matrix)
    # 데이터프레임에 'firms_list' 열 추가
    df_combined.insert(0, 'Firm', firms_list)
    df_combined.to_csv(output_file, index=False)

    
    # Print the download link and file path for the saved CSV file
    # df_sorted의 열은 "Firm", "Mom1", "LS", "Rank", "Cluster"
    download_link = urllib.parse.quote(output_file)
    file_path = os.path.abspath(output_file)
    print(f"Download link: {download_link}")
    print(f"File path: {file_path}")   

