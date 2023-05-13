import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

X_moon, y_moon = make_moons(n_samples=1000, noise=0.08, random_state=0)
X_blobs, y_blobs = make_blobs(n_samples=450, centers=[[0, 2], [2, 0], [0, -2]], cluster_std=0.4, n_features=2, random_state=0)

##############
#              SPECTRAL CLUSTERING
##############
from sklearn.cluster import SpectralClustering
from spectral import SpectralClustering as sc
fig, axes = plt.subplots(1, 2, figsize=(8, 5))

model = SpectralClustering(n_clusters=2, gamma=10)   # RBF - np.exp(-gamma * d(X, X) ** 2)
axes[0].scatter(X_moon[:, 0], X_moon[:, 1], c=model.fit_predict(X_moon))

model = sc(n_clusters=2, gamma=10, cut='normalized')   # RBF - np.exp(-gamma * d(X, X) ** 2)
axes[1].scatter(X_moon[:, 0], X_moon[:, 1], c=model.fit_predict(X_moon))
plt.show()

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
