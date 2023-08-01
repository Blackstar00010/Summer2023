from Hierarchical_Agglomerative import *
# 파일 불러오기
input_dir = '../files/PCA/PCA(1-48)_adj'
output_dir = '../files/Clustering_adj/Hierarchical_Agglomerative'
Hierarchical_Agglomerative = sorted(filename for filename in os.listdir(input_dir))

# CSV 파일 하나에 대해서 각각 실행
for file in Hierarchical_Agglomerative:

    data = read_and_preprocess_data(input_dir, file)

    clust=HG(data, 0.5)

    # 3. Save CSV
    columns = ['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index']
    new_table_generate(data, clust, output_dir, file)
