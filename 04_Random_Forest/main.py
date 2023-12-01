import numpy as np


class RandomForest:
    def __init__(self, n_trees, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        for _ in range(self.n_trees):
            # Randomly choose samples with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Create a decision tree and fit it
            tree = DecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X_bootstrap, y_bootstrap)

            # Add the tree to the forest
            self.trees.append(tree)

    def predict(self, X):
        # Make predictions by aggregating results from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Use majority voting for classification
        return np.mean(predictions, axis=0) > 0.5


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Base case: If only one class in the data or max depth reached
        if (
            len(unique_classes) == 1
            or (self.max_depth is not None and depth == self.max_depth)
            or n_samples < self.min_samples_split
        ):
            self.tree = unique_classes[0]
            return

        # Find the best split
        best_split = self.find_best_split(X, y)

        if best_split is not None:
            feature_index, threshold = best_split
            left_mask = X[:, feature_index] <= threshold
            right_mask = ~left_mask

            # Recursively build subtrees
            left_subtree = DecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            left_subtree.fit(X[left_mask], y[left_mask], depth + 1)

            right_subtree = DecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            right_subtree.fit(X[right_mask], y[right_mask], depth + 1)

            self.tree = (feature_index, threshold, left_subtree, right_subtree)

    def find_best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return None

        # Calculate Gini impurity for the current node
        current_gini = self.calculate_gini(y)

        best_gini = 1
        best_split = None

        # Iterate over all features
        for feature_index in range(n_features):
            # Sort the feature values and find midpoints
            feature_values = np.sort(np.unique(X[:, feature_index]))
            midpoints = (feature_values[:-1] + feature_values[1:]) / 2

            # Iterate over midpoints and find the best split
            for threshold in midpoints:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate Gini impurity for the split
                left_gini = self.calculate_gini(y[left_mask])
                right_gini = self.calculate_gini(y[right_mask])

                # Weighted sum of child node impurities
                weighted_gini = (
                    np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini
                ) / n_samples

                # Update the best split if the current one is better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = (feature_index, threshold)

        return best_split

    def calculate_gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def predict(self, X):
        if self.tree is not None:
            return np.array([self._predict(x, self.tree) for x in X])
        else:
            raise ValueError("Decision tree not fitted.")

    def _predict(self, x, node):
        if len(node) == 1:
            return node[0]  # Leaf node, return the class label
        else:
            feature_index, threshold, left_subtree, right_subtree = node
            if x[feature_index] <= threshold:
                return self._predict(x, left_subtree)
            else:
                return self._predict(x, right_subtree)


# Example usage:
# Assuming X_train, y_train, X_test are your training features, training labels, and testing features respectively.
# rf = RandomForest(n_trees=10, max_depth=5, min_samples_split=2)
# rf.fit(X_train, y_train)
# predictions = rf.predict(X_test)
