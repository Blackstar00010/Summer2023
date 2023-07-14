import os
import urllib.parse
import pandas as pd
from _gmm import *


# 1. 파일 불러오기
raw_data_dir = '../files/Clustering/PCA'
pca_output_dir = '../files/Clustering/Gaussian_Mixture_Model'
csv_files = [file for file in os.listdir(raw_data_dir) if file.endswith('.csv')]


# 2. CSV 파일 하나에 대해서 각각 실행
for file in csv_files:
    
    
    # 3. CSV파일 df변환
    csv_path = os.path.join(raw_data_dir, file)
    data = pd.read_csv(csv_path, header=None, index_col=[0])
    # 회사이름 추출 후 Value만 가지고 있는 dataframe생성.
    data=data[1:]
    # PCA 주성분 데이터만 가지고 있는 mata과 원본 Mom1을 추가로 가지고 있는 LS생성.
    LS=data.values
    mata=LS[0:,1:]
    mata = mata.astype(float)
    LS = LS.astype(float)

    
    # 4. GMM 알고리즘 구현
    DEBUG = True
    Y=mata
    matY = np.matrix(Y, copy=True)
    K = 4
    mu, cov, alpha = GMM_EM(matY, K, 100)
    N = Y.shape[0]
    gamma = getExpectation(matY, mu, cov, alpha)
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    
    # Cluster 4개 생성.
    class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
    class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
    class3 = np.array([Y[i] for i in range(N) if category[i] == 2])
    class4 = np.array([Y[i] for i in range(N) if category[i] == 3])
    
    # Cluster별 회사이름 할당.
    class_indices = []
    for i in range(K):
        indices = [data.index[index] for index, c in enumerate(category) if c == i][0:]
        class_indices.append(indices)
        
    # 회사별 mom1 할당.
    for i, indices in enumerate(class_indices):
        class_name = f"Class {i+1}"
        class_indices_dict[class_name] = indices
    cluster_elements = {i: [] for i in range(1, K+1)}
    
    for i in range(N):
        cluster = category[i]
        index = data.index[i]
        value = LS[i, 0]  # 첫 번째 열의 값
        cluster_elements[cluster+1].append((index, value))
        
        
    # 5. Outlier선별(예정)

        
    # 6. Result CSV 구현 및 생성
    output_file = os.path.join(pca_output_dir, file)
    df = pd.DataFrame(columns=['Firm', 'Value', 'Rank Value', 'Cluster'])
    for cluster, elements in cluster_elements.items():
        elements.sort(key=lambda x: x[1], reverse=True)
        num_elements = len(elements)
        median_index = num_elements // 2
        median_value = elements[median_index][1]
        for rank, (index, value) in enumerate(elements):
            if value > median_value:
                rank_value = 1
            elif value < median_value:
                rank_value = -1
            else:
                rank_value = 0
            df = pd.concat([df, pd.DataFrame({'Firm': [index], 'Value': [value], 'Rank Value': [rank_value], 'Cluster': [cluster]})])
    
    
    # 7. Print the download link and file path for the saved CSV file
    # df_sorted의 열은 "Firm", "Mom1", "LS", "Rank", "Cluster"
    df.to_csv(output_file, index=False)
    download_link = urllib.parse.quote(output_file)
    file_path = os.path.abspath(output_file)
    print(f"Download link: {download_link}")
    print(f"File path: {file_path}")