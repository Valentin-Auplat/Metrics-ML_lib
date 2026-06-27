import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Génération du dataset synthétique
X, y_true = make_blobs(
    n_samples=150,
    centers=3,
    cluster_std=2.5,
    random_state=42
)

# Visualisation des données brutes
plt.scatter(X[:, 0], X[:, 1], s=40)
plt.title("Synthetic 2D dataset for K-means")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(which="both")
plt.show()

# K-means "tout fait" : remplace compute_distances, assign_clusters,
# update_centroids et la boucle kmeans() en une seule classe
km = KMeans(n_clusters=3, max_iter=200, n_init=1, random_state=42)
labels = km.fit_predict(X)
centroids = km.cluster_centers_

# Distortion (équivalent de compute_distortion) = inertia_
print("Distortion (inertia):", km.inertia_)

# Visualisation finale (équivalent de plot_clusters)
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=40, alpha=0.7)
plt.scatter(centroids[:, 0], centroids[:, 1],
            s=300, marker="X", c="red", edgecolors="black", label="Centroids")
plt.title(f"K-means iteration {km.n_iter_}")
plt.legend()
plt.grid(True)
plt.show()