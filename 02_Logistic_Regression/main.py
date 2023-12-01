import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, num_features):
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

    def compute_cost(self, y, y_pred):
        m = len(y)
        cost = -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost

    def fit(self, X, y):
        m, num_features = X.shape
        self.initialize_parameters(num_features)

        for _ in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            dz = y_pred - y
            dw = 1 / m * np.dot(X.T, dz)
            db = 1 / m * np.sum(dz)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = self.compute_cost(y, y_pred)
            if _ % 100 == 0:
                print(f"Iteration {_}, Cost: {cost}")

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred > 0.5).astype(int)


# Example usage:
# Assume you have training data X_train and labels y_train
# Also, assume you have test data X_test

# Initialize and train the model
model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)


# Evaluate the model, for example using accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")
