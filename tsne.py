import numpy as np
from copy import deepcopy

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=0.5, n_iter=50):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate_ = learning_rate
        self.n_iter_ = n_iter
        self.embedding_ = None
        self.kl_divergence_ = None
        self.sigma = None

    def perp(self, X):
        arr = np.zeros(X.shape[0])
        for j in range(1, len(X)):
            arr[j] = np.exp(-sum((X[0] - X[j]) ** 2) / (2 * self.sigma ** 2))
        arr /= sum(arr)
        return 2 ** (-arr[1:].dot(np.log(arr[1:])))

    def fit(self, X):
        #                binary search for sigma
        a, b = 0, 1000
        self.sigma = (a + b) / 2
        while abs(self.perp(X) - self.perplexity) > 0.05:
            if self.perp(X) > self.perplexity:
                b = self.sigma
            else:
                a = self.sigma
            print(a, b)
            self.sigma = (a + b) / 2
            print(self.sigma)

        #            calculating real probabilities - p
        p = np.zeros((X.shape[0], X.shape[0]))
        for i in range(len(X)):
            for j in range(len(X)):
                if j != i:
                    p[i][j] = np.exp(-sum((X[i] - X[j]) ** 2) / (2 * self.sigma ** 2))
        p /= p.sum()
        p = (p + p.T) / (2 * X.shape[0])
        print(p.sum())

        #           inintializing mapping points
        self.embedding_ = np.random.multivariate_normal(mean=[0, 0], cov=(10 ** -4) * np.identity(2), size=X.shape[0])

        #           GRADIENT DESCENT
        q = np.zeros((X.shape[0], X.shape[0]))
        for _ in range(self.n_iter_):
            for i in range(len(X)):
                for j in range(len(X)):
                    if j != i:
                        q[i][j] = (1 + sum((self.embedding_[i] - self.embedding_[j]) ** 2)) ** -1
            q /= q.sum()
            print(q.sum())

            a = deepcopy(self.embedding_)
            self.kl_divergence = 0
            for i in range(a.shape[0]):
                deriv = 0
                for j in range(a.shape[0]):
                    if j != i:
                        self.kl_divergence += p[i][j] * np.log(p[i][j] / q[i][j])
                        deriv += (p[i][j] - q[i][j]) * (self.embedding_[i] - self.embedding_[j]) / (
                                    1 + sum((self.embedding_[i] - self.embedding_[j]) ** 2))
                a[i] = a[i] - self.learning_rate_ * 4 * deriv

            self.embedding_ = a
            print(self.kl_divergence)

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
