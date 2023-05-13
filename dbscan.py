import matplotlib.pyplot as plt
import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', verbose=False):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.n_clusters = -1
        self.core_sample_indices_ = []
        self.components_ = []
        self.n_outliers = None
        self.verbose = verbose

    def cluster_expanding(self, i, index1, X):
        if sum(np.linalg.norm(X - X[index1], axis=1) <= self.eps) >= self.min_samples:
            self.core_sample_indices_.append(index1)
            self.components_.append(X[index1])
            self.labels_[index1] = i
            arr = np.nonzero(np.linalg.norm(X - X[index1], axis=1) <= self.eps)[0]
            if self.verbose:
                print(f'{index1}: {arr}')
            for k in arr:
                if self.labels_[k] != i:
                    self.labels_[k] = i
                    self.cluster_expanding(i, k, X)
        elif self.labels_[index1] == -2:
            self.labels_[index1] = -1
            self.n_clusters -= 1
            if self.verbose:
                if sum(self.labels_ == -2) == 0:
                    print(f'X[{index1}] - Outlier')
                else:
                    print(f'X[{index1}] - Not a core point but still might be border point')

    def fit(self, X):
        self.labels_ = -2 * np.ones(X.shape[0])
        if self.metric == 'euclidean':
            while sum(self.labels_ == -2) != 0:
                indexes = np.nonzero(self.labels_ == -2)[0]
                self.n_clusters += 1
                index1 = np.random.choice(a=indexes, size=1)[0]
                if self.verbose:
                    print('#' * 100)
                    print('#' * 100)
                    print(f'Untouched data - {len(indexes)}\nOutliers at the moment - {sum(self.labels_ == -1)}')
                    print(f'Starting point for current cluster: {X[index1]} with index: {index1}\nLabel = {self.n_clusters}')
                    print('#' * 100)
                    print('Core point index: neighbors')
                self.cluster_expanding(self.n_clusters, index1, X)
            self.n_outliers = sum(self.labels_ == -1)
            self.n_clusters += 1
        elif self.metric == 'precomputed':
            if X.shape[0] == X.shape[1]:
                pass
            else:
                pass

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


