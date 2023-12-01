import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Calculate distances to all points in the training set
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

        # Get indices of k-nearest training data points
        k_indices = np.argsort(distances)[: self.k]

        # Get the labels of the k-nearest training data points
        k_nearest_labels = self.y_train[k_indices]

        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common


# Example usage:
# Assume X_train and y_train are your training data and labels, and X_test is your test data

# Example data
X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 5]])
y_train = np.array([0, 0, 1, 1])

X_test = np.array([[3, 2]])

# Create and train KNN classifier
knn = KNN(k=2)
knn.fit(X_train, y_train)

# Make predictions on the test data
predictions = knn.predict(X_test)

print("Predicted class:", predictions[0])
