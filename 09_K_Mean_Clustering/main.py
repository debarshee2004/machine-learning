import numpy as np


def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    K-means clustering algorithm.

    Parameters:
    - X: Input data, numpy array of shape (n_samples, n_features).
    - k: Number of clusters.
    - max_iters: Maximum number of iterations.
    - tol: Tolerance to declare convergence.

    Returns:
    - centroids: Final cluster centroids.
    - labels: Index of the cluster each sample belongs to.
    """
    # Initialize centroids randomly
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels


# Example usage:
np.random.seed(42)
# Generating some random data with two clusters
data = np.concatenate([np.random.randn(100, 2) + 5, np.random.randn(100, 2) - 5])

k = 2
centroids, labels = kmeans(data, k)

print("Final Centroids:")
print(centroids)
print("Labels:")
print(labels)
