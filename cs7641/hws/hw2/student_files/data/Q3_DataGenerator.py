import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

n_points = 600
n_clusters = 3
X, y = make_blobs(
    n_samples=n_points, centers=n_clusters, n_features=2, random_state=226
)
np.save("gaussian_clusters.npy", X)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
plt.title("Generated Gaussian Clusters")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
