import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.datasets import make_blobs

import os
#In case of retreiving a file to import functions (to modify depending on the architecture using os.path.dirname & os.path.join)
#sys.path.append(os.path.dirname(__file__), '..', 'file_location')
#from file_name import function

# #Retreiving data
# #paths definition
# path = os.path.dirname(__file__)
# data_path = os.path.join(path, "data_file")
# #data reading
# df = pd.read_csv(os.path.join(data_path, 'data_file_name'))

#On définit les fonctions qui permettent de générer, d'assigner et de mettre à jour les centroïdes :def compute_distances(X, centroids):


    # Generate a simple 2D dataset
X, y_true = make_blobs(
    n_samples=150,
    centers=3,
    cluster_std=2.5,
    random_state=42
)

# Plot it
plt.scatter(X[:, 0], X[:, 1], s=40)
plt.title("Synthetic 2D dataset for K-means")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(which="both")
plt.show()


def compute_distances(X, centroids):
    return np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=2)


def assign_clusters(X, centroids):
    
    distances = compute_distances(X, centroids)
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, K):

    n_features = X.shape[1]
    new_centroids = np.zeros((K, n_features))
    
    for k in range(K):
        cluster_points = X[labels == k]
        
        if len(cluster_points) > 0:
            new_centroids[k] = cluster_points.mean(axis=0)
        else:
            # Empty cluster: randomly reinitialize
            new_centroids[k] = X[np.random.choice(len(X))]
    
    return new_centroids


def compute_distortion(X, labels, centroids):

    return np.sum((X - centroids[labels])**2) / X.shape[0]


def plot_clusters(X, labels, centroids, iteration):

    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap="viridis", alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker="X")
    plt.title(f"K-means at iteration {iteration}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
    plt.grid(which="both")
    
    plt.show()


#On construit la fonction k-means :
def kmeans(X, K, max_iter=10):
    
    indices = np.random.choice(len(X), K, replace=False)
    centroids = X[indices]
    
    for iteration in range(max_iter):
        
        labels = assign_clusters(X, centroids)
        
        # Plot à chaque itération
        # plot_clusters(X, labels, centroids, iteration)
        
        centroids = update_centroids(X, labels, K)
    
    return labels, centroids

def plot_clusters(X, labels, centroids, iteration):
    
    plt.figure(figsize=(6, 5))
    
    plt.scatter(X[:, 0], X[:, 1],
                c=labels,
                cmap="viridis",
                s=40,
                alpha=0.7)
    
    plt.scatter(centroids[:, 0], centroids[:, 1],
                s=300,
                marker="X",
                c="red",
                edgecolors="black",
                label="Centroids")
    
    plt.title(f"K-means iteration {iteration}")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_clusters(X, kmeans(X, 3, 200)[0], kmeans(X, 3, 200)[1], 200)