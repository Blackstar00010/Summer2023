import matplotlib.pyplot as plt
from time import time
from _table_generate import *
from sklearn import datasets, manifold
from matplotlib.ticker import NullFormatter


def t_SNE(data, cluster):
    '''
    :param data: Mom data after PCA
    :param cluster: Original clutering data
    '''
    perplexities = [5, 30, 50, 100]
    clusters = cluster.fit_predict(data)

    # t-SNE를 사용하여 2차원으로 차원 축소

    for i in range(4):
        perplexity=perplexities[i]

        tsne = manifold.TSNE(
            n_components=2,
            random_state=0,
            perplexity=perplexity,
            learning_rate="auto",
            n_iter=1000,
        )

        X_tsne = tsne.fit_transform(data)
        plt.figure(figsize=(8, 6))
        plt.suptitle("Perplexity=%d" % perplexity)
        sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='plasma')

        plt.title('t-SNE Visualization')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')

        # 클러스터 라벨을 추가하여 범례(legend) 표시
        handles, labels = sc.legend_elements()
        plt.legend(handles, labels)
        plt.show()
        print()


if __name__ == "__main__":
    n_samples = 150
    n_components = 2
    (fig, subplots) = plt.subplots(4, 5, figsize=(15, 10))
    perplexities = [5, 30, 50, 100]

    first = False
    if first:
        X, y = datasets.make_circles(
            n_samples=n_samples, factor=0.5, noise=0.05, random_state=0
        )

        print(X)
        print(y)

        red = y == 0
        green = y == 1

        ax = subplots[0][0]
        ax.scatter(X[red, 0], X[red, 1], c="r")
        ax.scatter(X[green, 0], X[green, 1], c="g")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis("tight")

        for i, perplexity in enumerate(perplexities):
            ax = subplots[0][i + 1]

            t0 = time()
            tsne = manifold.TSNE(
                n_components=n_components,
                init="random",
                random_state=0,
                perplexity=perplexity,
                n_iter=300,
            )
            Y = tsne.fit_transform(X)
            t1 = time()
            print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
            ax.set_title("Perplexity=%d" % perplexity)
            ax.scatter(Y[red, 0], Y[red, 1], c="r")
            ax.scatter(Y[green, 0], Y[green, 1], c="g")
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis("tight")

    second = False
    if second:
        # Another example using s-curve
        X, color = datasets.make_s_curve(n_samples, random_state=0)

        ax = subplots[1][0]
        ax.scatter(X[:, 0], X[:, 1], c=color)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())

        for i, perplexity in enumerate(perplexities):
            ax = subplots[1][i + 1]

            t0 = time()
            tsne = manifold.TSNE(
                n_components=n_components,
                init="random",
                random_state=0,
                perplexity=perplexity,
                learning_rate="auto",
                n_iter=300,
            )
            Y = tsne.fit_transform(X)
            t1 = time()
            print("S-curve, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

            ax.set_title("Perplexity=%d" % perplexity)
            ax.scatter(Y[:, 0], Y[:, 1], c=color)
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis("tight")

    third = False
    if third:
        # Another example using a 2D uniform grid
        x = np.linspace(0, 1, int(np.sqrt(n_samples)))
        xx, yy = np.meshgrid(x, x)
        X = np.hstack(
            [
                xx.ravel().reshape(-1, 1),
                yy.ravel().reshape(-1, 1),
            ]
        )
        color = xx.ravel()
        ax = subplots[2][0]
        ax.scatter(X[:, 0], X[:, 1], c=color)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())

        for i, perplexity in enumerate(perplexities):
            ax = subplots[2][i + 1]

            t0 = time()
            tsne = manifold.TSNE(
                n_components=n_components,
                init="random",
                random_state=0,
                perplexity=perplexity,
                n_iter=400,
            )
            Y = tsne.fit_transform(X)
            t1 = time()
            print("uniform grid, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

            ax.set_title("Perplexity=%d" % perplexity)
            ax.scatter(Y[:, 0], Y[:, 1], c=color)
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis("tight")

        plt.show()
