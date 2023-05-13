import numpy as np
import matplotlib.pyplot as plt


def plot(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], marker='.', color='gray', label='data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1], color='black', label='previously selected centroids')
    plt.scatter(centroids[-1, 0], centroids[-1, 1], color='red', label='next centroid')
    plt.title('Select % d th centroid' % (centroids.shape[0]))
    plt.legend()
    plt.show()


class KMeans:
    def __init__(self, n_clusters, init='k-means++', max_iter=300, init_plot=False):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.init_plot = init_plot
        self.labels_ = None
        self.cluster_centers_ = None

    def init_centers(self, X):
        if self.init == 'random':
            return X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]
        if self.init == 'k-means++':
            centers = []
            f = X[np.random.randint(len(X))]
            centers.append(f)
            if self.init_plot:
                plot(X, np.array(centers))
            for j in range(self.n_clusters - 1):
                d = np.ones((len(centers), len(X)))
                for i, c in enumerate(centers):
                    d[i] = np.linalg.norm(X - c, axis=1)
                dist = d.min(axis=0)
                centers.append(X[np.random.choice(range(len(X)), p=dist / sum(dist))])
                if self.init_plot:
                    plot(X, np.array(centers))
            return np.array(centers)

    def fit(self, X, show=False):
        self.cluster_centers_ = self.init_centers(X)
        for _ in range(self.max_iter):
            self.labels_ = self.expectation(X, self.cluster_centers_)
            new_centroids = self.maximization(X, self.labels_)
            if (new_centroids == self.cluster_centers_).all():
                break
            if show:
                plt.scatter(X[:, 0], X[:, 1], c=self.labels_, s=50, cmap='viridis')
                plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='black', s=200, alpha=1)
                plt.show()
            self.cluster_centers_ = new_centroids

    def expectation(self, X, centroids):
        m, n = X.shape
        clusters = np.ones(m)
        for i, j in enumerate(X):
            a = np.linalg.norm(centroids - j, axis=1)
            clusters[i] = np.argmin(a)
        return clusters

    def maximization(self, X, clusters):
        new_centroids = []
        for i in range(self.n_clusters):
            new_centroids.append(X[clusters == i].mean(axis=0))
        return np.array(new_centroids)

    def predict(self, X):
        return self.expectation(X, self.cluster_centers_)

    def predict_proba(self, X):
        a = []
        for x in X:
            arr = 1 / np.linalg.norm(self.cluster_centers_ - x, axis=1)
            a.append(arr / (arr.sum()))
        return np.array(a)



