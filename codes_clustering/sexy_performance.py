from PCA_and_ETC import *

# turn off warning
warnings.filterwarnings("ignore")

MOM_merged_df = pd.read_csv('../files/mom1_data_combined_adj_close.csv')
MOM_merged_df.set_index('Firm Name', inplace=True)
MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)

base_directory = '../files/clustering_result/'

# Get all subdirectories in the base directory
subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

file_names = []
result_df = pd.DataFrame()
proportion = []

# Save subdir name in file_names at the beginning.
for subdir in subdirectories:
    # DBSCAN / Gaussian_Mixture_Model / HDBSCAN / Hierarchical_Agglomerative / ...
    file_names.append(subdir)

for subdir in subdirectories:
    print(subdir)
    directory = os.path.join(base_directory, subdir)

    LS_merged_df = pd.DataFrame()
    files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))
    for file in files:
        data = pd.read_csv(os.path.join(directory, file))
        LS_merged_df = merge_LS_Table(data, LS_merged_df, file)

    result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df)

save_and_plot_LS_Table(result_df, file_names, FTSE=True, apply_log=True)
