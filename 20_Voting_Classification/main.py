import numpy as np


class RandomForestClassifier:
    def __init__(
        self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.estimators = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Randomly select indices for bootstrap sampling
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(tree)

    def predict(self, X):
        # Use majority voting for predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        return np.mean(predictions, axis=0)


class VotingClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)

    def predict(self, X):
        # Use majority voting for predictions from all classifiers
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        return np.round(np.mean(predictions, axis=0))


# Example usage:
# Assuming you have your dataset in X_train, y_train, X_test
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ensemble of decision trees
rf_classifier1 = RandomForestClassifier(n_estimators=50, max_depth=3)
rf_classifier2 = RandomForestClassifier(n_estimators=50, max_depth=5)

# Create a voting classifier
voting_classifier = VotingClassifier(classifiers=[rf_classifier1, rf_classifier2])

# Train the voting classifier
voting_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = voting_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
