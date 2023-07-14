import os
import csv
import urllib.parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, cluster_optics_dbscan


# 1. 파일 불러오기
raw_data_dir = '../files/PCA'
pca_output_dir = '../files/Clustering/OPTICS'
csv_files = [file for file in os.listdir(raw_data_dir) if file.endswith('.csv')]


# 2. CSV 파일 하나에 대해서 실행
for file in csv_files:
    
    # 3. csv파일 df변환
    csv_path = os.path.join(raw_data_dir, file)
    data = pd.read_csv(csv_path, header=None)
    # 회사이름 추출 후 Value만 가지고 있는 dataframe생성.
    firm_lists = data[data.columns[0]].tolist()[1:]
    data = data.set_index(data.columns[0])
    data = data[1:]
    # PCA 주성분 데이터만 가지고 있는 mat과 원본 Mom1을 추가로 가지고 있는 LS생성.
    LS = data.values
    mat = LS[0:,1:]

    
    # 4. OPTICS 알고리즘 구현
    # xi = 거리, min_samples = 포함할 최소 데이터 수, min_cluster_size는 생성될 최소 군집 수
    clust = OPTICS(min_samples=3, xi=0.1, min_cluster_size=3)
    # 구분된 군집에 cluster_label부여하여 딕셔너리 형태로 저장.
    cluster_labels = clust.fit_predict(mat)
    unique_labels = set(cluster_labels)
    clusters = {label: [] for label in unique_labels}
    #Firm_lists의 index를 firm_index에 저장.
    for i, cluster_label in enumerate(cluster_labels):
        clusters[cluster_label].append(firm_lists[i])
    for cluster_label, firms in clusters.items():
        f'Cluster {cluster_label}: {firms}'
        for firm in firms:
            firm_index = list(firm_lists).index(firm)

            
    # 5. Result CSV 구현 및 생성.
    dat = pd.read_csv(csv_path, index_col=0) 
    unique_labels = set(label for label in cluster_labels if label != -1)
    output_file = os.path.join(pca_output_dir, file)
    LS_table = pd.DataFrame(columns=['Firm', 'Mom1', 'LS', 'Cluster'])
    clusters = {label: [] for label in unique_labels}
    for i, label in enumerate(cluster_labels):
        if label != -1:
            clusters[label].append(firm_lists[i])
    for cluster, firms in clusters.items():
        firms_sorted = sorted(firms, key=lambda x: dat.loc[x, '1'])
        long_short = [0] * len(firms_sorted)
        for i in range(len(firms_sorted) // 2):
            long_short[i] = -1  
            long_short[-i-1] = 1  
        for i, firm in enumerate(firms_sorted):
            LS_table.loc[len(LS_table)] = [firm, dat.loc[firm, '1'], long_short[i], cluster+1]

            
    # 6. Print the download link and file path for the saved CSV file
    # df_sorted의 열은 "Firm", "Mom1", "LS", "Rank", "Cluster"
    LS_table.to_csv(output_file, index=False)
    download_link = urllib.parse.quote(output_file)
    file_path = os.path.abspath(output_file)
    print(f"Download link: {download_link}")
    print(f"File path: {file_path}")