import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)

        # Check if only one class is present or maximum depth is reached
        if len(unique_classes) == 1 or (
            self.max_depth is not None and depth == self.max_depth
        ):
            return {"class": unique_classes[0]}

        # If no features are left, return the class with the majority
        if num_features == 0:
            majority_class = unique_classes[np.argmax(counts)]
            return {"class": majority_class}

        # Find the best feature and split point
        best_feature, split_point = self._find_best_split(X, y)

        if best_feature is None:
            # Unable to find a split, return the class with the majority
            majority_class = unique_classes[np.argmax(counts)]
            return {"class": majority_class}

        # Split the data based on the best feature and split point
        mask = X[:, best_feature] <= split_point
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        # Recursively build subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        # Return the node representing the decision
        return {
            "feature_index": best_feature,
            "split_point": split_point,
            "left": left_subtree,
            "right": right_subtree,
        }

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_feature = None
        best_split_point = None
        best_gini = 1.0  # Initialize with maximum Gini index

        for feature_index in range(num_features):
            feature_values = np.unique(X[:, feature_index])
            for split_point in feature_values:
                mask = X[:, feature_index] <= split_point
                y_left = y[mask]
                y_right = y[~mask]

                gini_left = self._calculate_gini(y_left)
                gini_right = self._calculate_gini(y_right)

                weighted_gini = (len(y_left) / num_samples) * gini_left + (
                    len(y_right) / num_samples
                ) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_split_point = split_point

        return best_feature, best_split_point

    def _calculate_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probability = counts / len(y)
        gini = 1.0 - np.sum(probability**2)
        return gini

    def predict(self, X):
        return np.array([self._predict_tree(sample, self.tree) for sample in X])

    def _predict_tree(self, sample, node):
        if "class" in node:
            return node["class"]
        else:
            if sample[node["feature_index"]] <= node["split_point"]:
                return self._predict_tree(sample, node["left"])
            else:
                return self._predict_tree(sample, node["right"])


# Example usage
if __name__ == "__main__":
    # Sample data
    X_train = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    y_train = np.array([0, 1, 0, 1])

    # Create and train the Decision Tree
    dt = DecisionTree(max_depth=2)
    dt.fit(X_train, y_train)

    # Predictions
    X_test = np.array([[0, 1], [1, 0]])
    predictions = dt.predict(X_test)

    print("Predictions:", predictions)
