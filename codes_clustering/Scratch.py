import Clustering as C
from PCA_and_tSNE import *
from sklearn.datasets import load_iris

iris=load_iris()

iris_pd = pd.DataFrame(iris.data[:,2:], columns=['petal_length', 'petal_width'])

# Plot K_mean cluster about individual csv file
example = True
if example:
    # Call initial method
    Do_Clustering = C.Clustering(iris_pd)

    # Do clustering and get 2D list of cluster index
    #Do_Clustering.K_Mean = Do_Clustering.perform_kmeans([10])
    Do_Clustering.DBSCAN=Do_Clustering.perform_DBSCAN()

    print(Do_Clustering.DBSCAN_labels)

    iris_pd['species']=iris.target
    # x_kc=Do_Clustering.test.cluster_centers_[:,0]
    # y_kc=Do_Clustering.test.cluster_centers_[:,1]

    import seaborn as sns

    plt.figure(figsize=(12,8))
    sns.scatterplot(x='petal_length', y='petal_width', hue='species',style='species', s=100, data=iris_pd)
    #plt.scatter(x_kc, y_kc, s=100, color='r')
    plt.scatter(iris_pd.iloc[:, 0], iris_pd.iloc[:, 1], c=Do_Clustering.test)
    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    plt.show()




