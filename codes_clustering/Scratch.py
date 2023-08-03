import seaborn as sns
import Clustering as C
from PCA_and_tSNE import *
from sklearn.datasets import load_iris
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

iris = load_iris()
iris_pd = pd.DataFrame(iris.data[:, 2:], columns=['petal_length', 'petal_width'])

# Plot K_mean cluster about individual csv file
example1 = False
if example1:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([3])

    iris_pd['species'] = iris.target
    x_kc = Do_Clustering.test.cluster_centers_[:, 0]
    y_kc = Do_Clustering.test.cluster_centers_[:, 1]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='petal_length', y='petal_width', hue='species', style='species', s=100, data=iris_pd)
    plt.scatter(x_kc, y_kc, s=100, color='r')
    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    plt.show()

    t_SNE(Do_Clustering.PCA_Data, Do_Clustering.K_Mean_labels)

example2 = False
if example2:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.DBSCAN = Do_Clustering.perform_DBSCAN2()

    n_clusters_ = len(set(Do_Clustering.DBSCAN_labels)) - (1 if -1 in Do_Clustering.DBSCAN_labels else 0)
    n_noise_ = list(Do_Clustering.DBSCAN_labels).count(-1)

    unique_labels = set(Do_Clustering.DBSCAN_labels)
    core_samples_mask = np.zeros_like(Do_Clustering.DBSCAN_labels, dtype=bool)
    core_samples_mask[Do_Clustering.test.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = Do_Clustering.DBSCAN_labels == k

        xy = Do_Clustering.PCA_Data[class_member_mask & core_samples_mask]
        plt.plot(
            xy.iloc[:, 0],
            xy.iloc[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = Do_Clustering.PCA_Data[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy.iloc[:, 0],
            xy.iloc[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

    t_SNE(Do_Clustering.PCA_Data, Do_Clustering.DBSCAN_labels)

example3 = True
if example3:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.OPTIC = Do_Clustering.perform_OPTICS()


    t_SNE(Do_Clustering.PCA_Data, Do_Clustering.OPTIC_labels)