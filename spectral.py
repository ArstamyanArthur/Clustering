import numpy as np
from kmeans import KMeans as km
from sklearn.cluster import KMeans
from scipy.linalg import eigh


class SpectralClustering:
    def __init__(self, n_clusters=2, affinity='rbf', n_neighbors=10, n_init=10, gamma=1, cut='unnormalized'):
        self.L = None
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.D = None
        self.cut = cut

    def fit(self, X):
        W = np.zeros((X.shape[0], X.shape[0]))
        self.D = np.zeros((X.shape[0], X.shape[0]))
        if self.affinity == 'rbf':
            for i in range(len(X)):
                for j in range(len(X)):
                    W[i][j] = np.exp(-self.gamma * sum((X[i]-X[j])**2))
                self.D[i][i] = sum(W[i])
        elif self.affinity == 'nearest_neighbors':
            for i in range(len(X)):
                for j in np.argsort(np.linalg.norm(X-X[i], axis=1))[:self.n_neighbors]:
                    W[i][j] = 1
                self.D[i][i] = sum(W[i])
            W = (W + W.T)/2
        self.L = self.D - W

    def fit_predict(self, X):
        self.fit(X)
        if self.cut == 'normalized':
            v, u = eigh(self.L, self.D)
        else:
            v, u = eigh(self.L)
        u = u[:, :self.n_clusters]
        m = KMeans(n_clusters=self.n_clusters, n_init=self.n_init)
        return m.fit_predict(u)


