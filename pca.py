import numpy as np
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components_ = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        self.components_ = np.linalg.eigh(cov)[1][:, -self.n_components:][:, ::-1].T

    def transform(self, X):
        return X.dot(self.components_.T)
