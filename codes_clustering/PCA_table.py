from _table_generate import *
from sklearn.decomposition import PCA
from PCA_single import *

# 파일 불러오기
input_dir = '../files/momentum_adj'
output_dir = '../files/PCA/PCA(1-48)_adj'
momentum = sorted(filename for filename in os.listdir(input_dir))

# CSV 파일 하나에 대해서 각각 실행.
for file in momentum:
    if file == '.DS_Store':
        continue

    data = read_and_preprocess_data(input_dir, file)
    df_combined=generate_PCA_File(data)
    print(file)

    # 4. Save CSV
    # Column format: ['Original Mom1', 'data after PCA', ...]
    output_file = os.path.join(output_dir, file)
    df_combined.to_csv(output_file, index=True)
