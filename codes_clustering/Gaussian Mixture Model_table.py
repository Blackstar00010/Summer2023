from Gaussian_Mixture_Model import *

# 파일 불러오기
input_dir = '../files/PCA/PCA(1-48)_adj'
output_dir = '../files/Clustering_adj/Gaussian_Mixture_Model'
Gaussian_Mixture_Model = sorted(filename for filename in os.listdir(input_dir))


# CSV 파일 하나에 대해서 각각 실행
for file in Gaussian_Mixture_Model:
    data = read_and_preprocess_data(input_dir, file)

    # if data are lower than 5, using GMM is impossible.
    if len(data) < 5:
        continue


    clusters=GMM(data, 0.1)

    # 3. Save CSV
    # columns = ['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index']
    new_table_generate(data, clusters, output_dir, file)
