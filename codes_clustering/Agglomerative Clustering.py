import matplotlib.pyplot as plt
from _table_generate import *
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# 데이터 불러오기
input_dir = '../files/PCA/PCA(1-48)'
file = '2016-01.csv'
data = read_and_preprocess_data(input_dir, file)
mat = data.values[:, 1:].astype(float)

model = AgglomerativeClustering(n_clusters=10, linkage='ward', distance_threshold=None)

model = model.fit(mat)

cluster_labels = model.fit_predict(mat)

model

print(cluster_labels)

first = False
if first:
    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)


    plt.title("Hierarchical Clustering Dendrogram")
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()