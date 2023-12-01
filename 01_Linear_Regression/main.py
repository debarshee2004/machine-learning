import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Add a column of ones to X for the bias term
        X = np.c_[np.ones(X.shape[0]), X]

        # Initialize weights and bias to zeros
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            predictions = self.predict(X)
            errors = predictions - y

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * (1 / X.shape[0]) * np.dot(X.T, errors)
            self.bias -= self.learning_rate * (1 / X.shape[0]) * np.sum(errors)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Example usage:
# Assuming you have your data in variables X and y
# X should be a 2D array where each row is a data point and each column is a feature
# y should be a 1D array of the corresponding target values

# Example data:
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 5])

# Create and train the model
model = LinearRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(np.c_[np.ones(X.shape[0]), X])

print("Weights:", model.weights)
print("Bias:", model.bias)
print("Predictions:", predictions)
