import numpy as np


class MultinomialNaiveBayes:
    def __init__(self):
        self.class_probabilities = None
        self.feature_probabilities = None

    def fit(self, X, y):
        num_classes = np.max(y) + 1
        num_features = X.shape[1]

        # Calculate class probabilities
        self.class_probabilities = np.zeros(num_classes)
        for i in range(num_classes):
            self.class_probabilities[i] = np.sum(y == i) / len(y)

        # Calculate feature probabilities
        self.feature_probabilities = np.zeros((num_classes, num_features))
        for i in range(num_classes):
            class_samples = X[y == i]
            self.feature_probabilities[i] = (np.sum(class_samples, axis=0) + 1) / (
                np.sum(class_samples) + num_features
            )

    def predict(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.class_probabilities)

        # Log likelihoods for each class
        log_likelihoods = np.zeros((num_samples, num_classes))

        for i in range(num_classes):
            log_likelihoods[:, i] = np.sum(
                np.log(self.feature_probabilities[i]) * X, axis=1
            ) + np.log(self.class_probabilities[i])

        # Predict the class with the highest log likelihood
        predictions = np.argmax(log_likelihoods, axis=1)

        return predictions


# Example usage:
# Assuming X_train and y_train are your training data
X_train = np.array([[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 0, 1], [0, 0, 1, 1]])

y_train = np.array([0, 1, 0, 1])

# Assuming X_test is your test data
X_test = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])

nb_classifier = MultinomialNaiveBayes()
nb_classifier.fit(X_train, y_train)
predictions = nb_classifier.predict(X_test)

print("Predictions:", predictions)
