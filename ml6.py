# K-means clustering algorithm using the popular scikit-learn library.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic data for clustering
# We will create 300 samples, each with 2 features, grouped into 4 clusters
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Visualizing the synthetic data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Data before clustering")
plt.show()

# Implementing K-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Visualizing the clustered data with centroids
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title("Data after K-means clustering")
plt.legend()
plt.show()

