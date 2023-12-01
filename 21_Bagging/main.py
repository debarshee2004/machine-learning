import numpy as np


class BaggingClassifier:
    def __init__(self, base_model, n_estimators):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Randomly sample with replacement (bootstrap)
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Train base model on the bootstrap sample
            model = self.base_model()
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)

    def predict(self, X):
        # Use majority voting for classification
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0).round().astype(int)


# Example usage with a simple decision tree as the base model
class SimpleDecisionTree:
    def __init__(self):
        pass  # Implement your decision tree here

    def fit(self, X, y):
        pass  # Implement the training of your decision tree

    def predict(self, X):
        pass  # Implement the prediction of your decision tree


# Example usage:
# Create a dataset for testing
np.random.seed(42)
X_train = np.random.rand(100, 5)
y_train = (X_train.sum(axis=1) > 2.5).astype(int)

X_test = np.random.rand(10, 5)

# Create BaggingClassifier with SimpleDecisionTree as the base model
bagging_classifier = BaggingClassifier(base_model=SimpleDecisionTree, n_estimators=10)

# Fit the ensemble model
bagging_classifier.fit(X_train, y_train)

# Make predictions
predictions = bagging_classifier.predict(X_test)

print("Predictions:", predictions)
