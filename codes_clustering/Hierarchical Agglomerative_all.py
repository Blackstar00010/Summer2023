#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import csv
import urllib.parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
get_ipython().run_line_magic('matplotlib', 'inline')


#raw_data_dir = 'C:/Users/김주환/Desktop/My files/PCA'
#pca_output_dir = 'C:/Users/김주환/Desktop/My files/Hierarchical_Agglomerative'
raw_data_dir = 'C:/Users/IE/Desktop/My files/PCA'
pca_output_dir = 'C:/Users/IE/Desktop/My files/Hierarchical_Agglomerative'


# Get a list of all CSV files in the raw data directory
csv_files = [file for file in os.listdir(raw_data_dir) if file.endswith('.csv')]


# Loop through each CSV file
for file in csv_files:
    # Read the CSV file
    csv_path = os.path.join(raw_data_dir, file)
    data = pd.read_csv(csv_path, header=None)
    
    firms_list = data[data.columns[0]].tolist()[1:]
    data = data.set_index(data.columns[0])
    data = data[1:]
    
    mat = data.values.astype(float)
    LS=data.values
    mat=LS[0:,1:]


    # 거리 행렬 계산
    dist_matrix = pdist(mat, metric='euclidean')

    # 연결 매트릭스 계산
    Z = linkage(dist_matrix, method='ward')

    # 덴드로그램 시각화
    dendrogram(Z)

#     plt.title('Dendrogram')
#     plt.xlabel('Samples')
#     plt.ylabel('Distance')
#     plt.show()


    # 적절한 클러스터 개수 선택
    # 덴드로그램을 분석하여 적절한 클러스터 개수를 결정합니다.

    # 클러스터 할당
    k = 100  # 예시로 클러스터 개수를 3으로 설정
    clusters = fcluster(Z, k, criterion='maxclust')




    #Cluster별 분류
    list_dict = {}
    for i in range(k):
         list_dict[i+1] = []

    for i in range(0,len(firms_list)):
        for j in range(0,k):
            if clusters[i]==j+1:
              list(list_dict.values())[j].append('firm'+str(i+1)+': '+str(LS[0:,0][i]))

    for key, value in list_dict.items():
        print(value)
        print(key)




    #LS와 Rank
    for key, value in list_dict.items():
        print(f"Cluster: {key}")
        sorted_data = sorted(value, key=lambda x: float(x.split(': ')[1]), reverse=True)
        length = len(sorted_data)
        for i, item in enumerate(sorted_data):
            rank = i - length // 2
            if length%2==0:
              if rank == 0:
                h=int((length)/2)+1
                t=1


              if rank > 0:
                h=rank+int((length)/2)+1
                t=1

              elif rank < 0:
                h=rank+int((length)/2)+1
                t=-1

              print(f"{item}: {t}: {h}")

            else:
              if rank == 0:
                h=rank+int((length-1)/2)+1
                t=0

              if rank > 0:
                h=rank+int((length-1)/2)+1
                t=1

              elif rank < 0:
                h=rank+int((length-1)/2)+1
                t=-1

              #print(f"{item}: {t}: {h}")



    #행렬로 변환
    result_matrix = []

    for key, value in list_dict.items():
        row = [key]  # Cluster 값을 첫 번째 열에 저장
        sorted_data = sorted(value, key=lambda x: float(x.split(': ')[1]), reverse=True)
        length = len(sorted_data)
        for i, item in enumerate(sorted_data):
            rank = i - length // 2
            if length%2==0:
              if rank == 0:
                h=int((length)/2)+1
                t=1


              if rank > 0:
                h=rank+int((length)/2)+1
                t=1

              elif rank < 0:
                h=rank+int((length)/2)+1
                t=-1

              row.append(f"{item}: {t}: {h}")

            else:
              if rank == 0:
                h=rank+int((length-1)/2)+1
                t=0

              if rank > 0:
                h=rank+int((length-1)/2)+1
                t=1

              elif rank < 0:
                h=rank+int((length-1)/2)+1
                t=-1

              row.append(f"{item}: {t}: {h}")



        result_matrix.append(row)

    # 결과 행렬 출력
    for row in result_matrix:
        print(row)




    #행렬 csv로 변환

    result_matrix = []

    for key, value in list_dict.items():
        row = [key]  # Cluster 값을 첫 번째 열에 저장
        sorted_data = sorted(value, key=lambda x: float(x.split(': ')[1]), reverse=True)
        length = len(sorted_data)
        for i, item in enumerate(sorted_data):
            rank = i - length // 2
            if length%2==0:
              if rank == 0:
                h=int((length)/2)+1
                t=1


              if rank > 0:
                h=rank+int((length)/2)+1
                t=1

              elif rank < 0:
                h=rank+int((length)/2)+1
                t=-1

              row.append(f"{item}: {t}: {h}")

            else:
              if rank == 0:
                h=rank+int((length-1)/2)+1
                t=0

              if rank > 0:
                h=rank+int((length-1)/2)+1
                t=1

              elif rank < 0:
                h=rank+int((length-1)/2)+1
                t=-1

              row.append(f"{item}: {t}: {h}")


        result_matrix.append(row)  # 행을 결과 행렬에 추가

        
    # 결과 행렬을 DataFrame으로 변환
    df_result = pd.DataFrame(result_matrix)
    df = df_result


    #맨 뒤에 소속 Cluster 붙이기
    for i in range(0,100):
      for j in range(0,len(df.head(i+1).iloc[i].dropna())-1):
        df.iloc[i, j+1] = str(df.iloc[i, j+1]) + ':' + str(df.iloc[i, 0])

    df = df.iloc[:, 1:]


    #data frame into list
    result_list = []

    for row in df.values:
        for item in row:
            if pd.notnull(item):  # 결측값이 아닌 경우에만 리스트에 추가
                result_list.append(item)


    #회사 기준 정렬
    sorted_result_list = sorted(result_list, key=lambda x: int(x.split(":")[0][4:]))
    # ":" 기준 split
    split_result_list = [item.split(":") for item in sorted_result_list]

    
    # Save the PCA results as CSV in the PCA output directory
    output_file = os.path.join(pca_output_dir, file)
    df_sorted = pd.DataFrame(split_result_list)
    df_sorted.columns=["Firm", "mom1", "LS", "Rank", "Cluster"]
    df_sorted = df_sorted.iloc[:, 1:]
    df_sorted.insert(0, 'firms_list', firms_list)


    # 데이터프레임을 CSV 파일로 저장
    df_sorted.to_csv(output_file, index=False)

    # 다운로드 링크와 파일 경로 출력
    download_link = urllib.parse.quote(output_file)
    file_path = os.path.abspath(output_file)
    print(f"Download link: {download_link}")
    print(f"File path: {file_path}")

