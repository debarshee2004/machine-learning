import numpy as np


class LabelPropagation:
    def __init__(self, n_neighbors=1, max_iter=1000, tol=1e-3):
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self.labels = None

    def fit(self, X, y):
        n_samples, _ = X.shape
        self.labels = np.copy(y)

        for _ in range(self.max_iter):
            # Propagate labels
            self._propagate_labels(X)

            # Check convergence
            changes = np.sum(self.labels != y)
            if changes < self.tol:
                break

        return self

    def _propagate_labels(self, X):
        n_samples, _ = X.shape

        for i in range(n_samples):
            neighbors_indices = self._find_neighbors(X, i)
            neighbor_labels = self.labels[neighbors_indices]

            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            most_common_label = unique_labels[np.argmax(counts)]

            self.labels[i] = most_common_label

    def _find_neighbors(self, X, index):
        distances = np.sum((X - X[index]) ** 2, axis=1)
        neighbors_indices = np.argsort(distances)[: self.n_neighbors]
        return neighbors_indices


# Example usage:
# Assuming X is your feature matrix and y is your initial labeled data
# X should contain both labeled and unlabeled data

# Generate a synthetic dataset for demonstration
np.random.seed(42)
X_labeled = np.random.rand(10, 2)
y_labeled = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1])

X_unlabeled = np.random.rand(90, 2)

X = np.concatenate((X_labeled, X_unlabeled))
y = np.concatenate((y_labeled, np.full(90, -1)))  # -1 represents unlabeled

# Create and fit the LabelPropagation model
model = LabelPropagation(n_neighbors=5, max_iter=1000, tol=1e-3)
model.fit(X, y)

# Get the final labels assigned by the algorithm
final_labels = model.labels

# Print the final labels
print("Final Labels:", final_labels)
