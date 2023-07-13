import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster


#데이터 불러오기
#data = pd.read_csv('C:/Users/김주환/Desktop/My files/PCA/2018-01.csv', header=None)
data = pd.read_csv('C:/Users/IE/Desktop/My files/PCA/2018-01.csv', header=None)
firms_list = data[data.columns[0]].tolist()[1:]
data = data.set_index(data.columns[0])
data=data[1:]
LS=data.values
mat=LS[0:,1:]
print('Mom1+PCA')
print(LS)
print('Only PCA')
print(mat)


#Hierarchical Agglomertive 구현
# 거리 행렬 계산
dist_matrix = pdist(mat, metric='euclidean')
# 연결 매트릭스 계산
Z = linkage(dist_matrix, method='ward')
# 덴드로그램 시각화
dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
# 클러스터 할당
k = 100  # 예시로 클러스터 개수를 3으로 설정
clusters = fcluster(Z, k, criterion='maxclust')
# 클러스터 할당 결과 출력
print(dist_matrix)
print("클러스터 할당 결과:")
print(clusters)


# Result CSV 구현
list_dict = {}
# 클러스터 갯수만큼 key생성
for i in range(k):
     list_dict[i+1] = []
# n번째 회사가 k번째 클러스터에 있으면 list_dict에 'firm_n'으로 value로 저장.
for i in range(0,len(firms_list)):
    for j in range(0,k):
        if clusters[i]==j+1:
          list(list_dict.values())[j].append('firm'+str(i+1)+': '+str(LS[0:,0][i]))
# 클러스터 내부에서 Frim Number 기준정렬 후 Rank와 Long Short Value계산하여result_matrix저장.
result_matrix = []

for key, value in list_dict.items():
    row = [key]
    sorted_data = sorted(value, key=lambda x: float(x.split(': ')[1]), reverse=True)
    length = len(sorted_data)
    # h = Rank, t = Long Short Value
    
     for i, item in enumerate(sorted_data):
        rank = i - length // 2
        h = rank + (length + 1) // 2
        t = 0 if length % 2 == 0 else (1 if rank > 0 else -1)
        result = f"{item}: {t}: {h}"
        row.append(result)
    result_matrix.append(row)

df = pd.DataFrame(result_matrix)

#df의 각 원소에 대하여 맨 뒤에 소속 Cluster를 붙임.
for i in range(0,100):
    for j in range(0,len(df.head(i+1).iloc[i].dropna())-1):
        df.iloc[i, j+1] = str(df.iloc[i, j+1]) + ':' + str(df.iloc[i, 0])

df = df.iloc[:, 1:]

# 결측값을 제거하고 pure data만 result_list에 저장
result_list = []

for row in df.values:
    for item in row:
        if pd.notnull(item):  
            result_list.append(item)

#list 값 정렬 후 ":"기준 분리
sorted_result_list = sorted(result_list, key=lambda x: int(x.split(":")[0][4:]))
split_result_list = [item.split(":") for item in sorted_result_list]


# Result CSV 생성    
#Firm Number 대신에 실제 Firm이름으로 대체.
df_sorted = pd.DataFrame(split_result_list)
df_sorted.columns=["Firm", "Mom1", "LS", "Rank", "Cluster"]
df_sorted = df_sorted.iloc[:, 1:]
df_sorted.insert(0, 'Firm', firms_list)
# 회사별로 정렬된 dataframe을 다시 Cluster에 대하여 정렬.
df_sorted[df_sorted.columns[4]] = pd.to_numeric(df_sorted[df_sorted.columns[4]])
df_sorted = df_sorted.sort_values(by=df_sorted.columns[4])
df_sorted = df_sorted.sort_values(by=['Cluster', df_sorted.columns[3]])
df_sorted


#CSV 받기
# import os
# import csv
# import urllib.parse
# #output_file = 'C:/Users/김주환/Desktop/My files/Hierarchical Agglomerative/2018-01.csv'
# output_file = 'C:/Users/IE/Desktop/My files/Hierarchical Agglomerative/2018-01.csv'
# df_sorted.to_csv(output_file, index=False)
# download_link = urllib.parse.quote(output_file)
# file_path = os.path.abspath(output_file)
# print(f"Download link: {download_link}")
# print(f"File path: {file_path}")
