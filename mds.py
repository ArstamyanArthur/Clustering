from sklearn.metrics import pairwise_distances
import numpy as np


class MDS:
    def __init__(self, n_components=2, max_iter=300, dissimilarity='euclidean'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.dissimilarity = dissimilarity
        self.dissimilarity_matrix = None
        self.stress_ = None
        self.embedding_ = None
        self.norm = None
        self.lr = 0.1

    def fit(self, X):
        self.dissimilarity_matrix = pairwise_distances(X)
        self.norm = np.linalg.norm(self.dissimilarity_matrix)

        self.embedding_ = 100 * np.random.rand(X.shape[0], self.n_components)
        D = pairwise_distances(self.embedding_)
        self.stress_ = np.linalg.norm(self.dissimilarity_matrix - D) / self.norm

        a = np.random.rand(X.shape[0], self.n_components)
        for _ in range(self.max_iter):
            for i in range(X.shape[0]):
                for j in range(self.n_components):
                    a[i, j] = self.embedding_[i, j] + self.lr * (1 / (2 * self.stress_)) * sum([2 * (
                                self.dissimilarity_matrix[i, k] - D[i, k]) * (self.embedding_[i, j] - self.embedding_[
                        k, j]) / D[i, k] for k in range(X.shape[0]) if k != i])
            self.embedding_ = a
            D = pairwise_distances(self.embedding_)
            self.stress_ = np.linalg.norm(self.dissimilarity_matrix - D)
            print(self.stress_)

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
