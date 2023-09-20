import pandas as pd
import Clustering as cl
import numpy as np
from PCA_and_ETC import *
from sklearn.metrics import silhouette_score

'''
perform_clustering + sexy_performance
'''


def kmeans_cluster(input_dir: str, print_logs=False, print_result=False):
    """

    :param input_dir:
    :param print_logs:
    :param print_result:
    :return:
    """
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    sil = 0
    cl = 0

    result_dict = []
    for afile in files:
        if print_logs:
            print(afile)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, afile)
        df_combined = generate_PCA_Data(data)

        # initialise Clustering class
        Do_Clustering = cl.Clustering(df_combined)
        Do_Result_Save = cl.ResultCheck(df_combined)

        # perform clustering and get 2D list of cluster index
        Do_Clustering.perform_HDBSCAN(0.5)

        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

        # Save LS_Table CSV File
        result_dict.append(Do_Result_Save.ls_table(Do_Clustering.HDBSCAN, '', afile, save=False)[['Firm Name', 'Long Short']].set_index('Firm Name'))

        if print_logs:
            print("Number of clusters is:", len(set(Do_Clustering.HDBSCAN_labels)))

    result_df = pd.concat(result_dict, axis=1)

    sil = sil / len(files)
    cl = cl / len(files)
    if print_result:
        print('average number of clusters:', cl)
        print('silhouette score:', sil)
        print(f'total outliers: {outliers_count}')


if __name__ == '__main__':
    df = pd.read_csv('../files/_test/result_modified.csv')
    df = df.rename(columns={'Unnamed: 0': 'method'}).transpose()
    df.columns = df.iloc[0]
    df.index = df.index.str[:7]
    df = df.drop(df.index[0])

    thresh = 0.1
    df = df * (df.abs() <= thresh) + (df > thresh) * thresh + (df < -thresh) * (-thresh)
    df = df + 1
    for amethod in df.columns:
        if amethod == 'Meanshift':
            amethod = 'Meanshift'
        vector = df[amethod] + 1
        vector = vector.cumprod()
        plt.plot(vector, label=amethod)
        plt.show()
        plt.title(amethod)
    df = df.cumprod()
    # df = np.log(df.astype(float) * 1.0)
    df.plot()
    plt.title('threhshold: ' + str(thresh))
    plt.figure(figsize=(10, 6))
    plt.show()
