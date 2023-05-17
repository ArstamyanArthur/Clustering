import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# X_moon, y_moon = make_moons(n_samples=1000, noise=0.08, random_state=0)
# X_blobs, y_blobs = make_blobs(n_samples=450, centers=[[0, 2], [2, 0], [0, -2]], cluster_std=0.4, n_features=2, random_state=0)

###########################
#              SPECTRAL CLUSTERING
###########################
# from sklearn.cluster import SpectralClustering
# from spectral import SpectralClustering as sc
# fig, axes = plt.subplots(1, 2, figsize=(8, 5))
#
# model = SpectralClustering(n_clusters=2, gamma=10)   # RBF - np.exp(-gamma * d(X, X) ** 2)
# axes[0].scatter(X_moon[:, 0], X_moon[:, 1], c=model.fit_predict(X_moon))
#
# model = sc(n_clusters=2, gamma=10, cut='normalized')   # RBF - np.exp(-gamma * d(X, X) ** 2)
# axes[1].scatter(X_moon[:, 0], X_moon[:, 1], c=model.fit_predict(X_moon))
# plt.show()

# fig, axes = plt.subplots(1, 2, figsize=(8, 5))
#
# model = SpectralClustering(n_clusters=3, gamma=30)
# axes[0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=model.fit_predict(X_blobs))
#
# model = sc(n_clusters=3, gamma=30, cut='normalized')
# axes[1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=model.fit_predict(X_blobs))
# plt.show()

################################
#                   DBSCAN
################################
# from sklearn.cluster import DBSCAN
# from dbscan import DBSCAN as db
#
# fig, axes = plt.subplots(1, 2, figsize=(8, 5))
#
# model = DBSCAN(eps=0.1, min_samples=5)
# axes[0].scatter(X_moon[:, 0], X_moon[:, 1], c=model.fit_predict(X_moon))
#
# model = db(eps=0.1, min_samples=5, verbose=False)
# axes[1].scatter(X_moon[:, 0], X_moon[:, 1], c=model.fit_predict(X_moon))
# plt.show()
# print(model.n_outliers, model.n_clusters, model.components_)

################################
#                   TSNE
################################

from tsne import TSNE
# from sklearn.manifold import TSNE
digits = datasets.load_digits()
X = digits.data[:200, :]
model = TSNE(perplexity=25, learning_rate=0.5)
N = model.fit_transform(X)
plt.scatter(N[:, 0], N[:, 1], c=digits.target[:200])
plt.show()

################################
#                   PCA
################################
from pca import PCA
# from sklearn.decomposition import PCA
digits = datasets.load_digits()
X = digits.data[:200, :]

pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

plt.scatter(X[:, 0], X[:, 1], c=digits.target[:200])
plt.show()


################################
#                  MDS
################################
from mds import MDS
# from sklearn.manifold import MDS
digits = datasets.load_digits()
X = digits.data[:200, :]

model = MDS()
N = model.fit_transform(X)
plt.scatter(N[:, 0], N[:, 1], c=digits.target[:200])
plt.show()

