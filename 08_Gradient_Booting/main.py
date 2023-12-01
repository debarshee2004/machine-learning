import numpy as np


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # Initialize the prediction with the mean of the target values
        initial_prediction = np.mean(y)
        prediction = np.full_like(y, initial_prediction)

        for _ in range(self.n_estimators):
            # Calculate the residuals
            residuals = y - prediction

            # Fit a weak learner (decision tree) to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update the prediction using the learning rate and the weak learner's prediction
            update = self.learning_rate * tree.predict(X)
            prediction += update

            # Save the weak learner for later use in predictions
            self.trees.append(tree)

    def predict(self, X):
        # Make predictions by summing the predictions of all weak learners
        return np.sum([tree.predict(X) for tree in self.trees], axis=0)


class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or np.all(y == y[0]):
            # Leaf node, return the mean of the target values
            return np.mean(y)

        # Find the best split
        split_index, split_value = self._find_best_split(X, y)

        if split_index is None:
            # No further split is beneficial
            return np.mean(y)

        # Split the data
        left_mask = X[:, split_index] <= split_value
        right_mask = ~left_mask

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return a node representing the split
        return (split_index, split_value, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        best_split_index, best_split_value, best_loss = None, None, float("inf")

        for i in range(X.shape[1]):
            # Sort the feature values and find potential split points
            unique_values = np.unique(X[:, i])
            split_points = (unique_values[:-1] + unique_values[1:]) / 2

            for split_value in split_points:
                left_mask = X[:, i] <= split_value
                right_mask = ~left_mask

                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    # Calculate the mean squared error loss
                    left_loss = np.mean((y[left_mask] - np.mean(y[left_mask])) ** 2)
                    right_loss = np.mean((y[right_mask] - np.mean(y[right_mask])) ** 2)
                    total_loss = left_loss + right_loss

                    # Update the best split if the current one is better
                    if total_loss < best_loss:
                        best_loss = total_loss
                        best_split_index = i
                        best_split_value = split_value

        return best_split_index, best_split_value

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if isinstance(node, float):
            # Leaf node, return the predicted value
            return node

        split_index, split_value, left_subtree, right_subtree = node

        if x[split_index] <= split_value:
            # Recur on the left subtree
            return self._predict_tree(x, left_subtree)
        else:
            # Recur on the right subtree
            return self._predict_tree(x, right_subtree)


# Example usage:
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + np.random.normal(scale=0.1, size=100)

gb_regressor = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3
)
gb_regressor.fit(X, y)

new_X = np.array([[0.5]])
predicted_y = gb_regressor.predict(new_X)

print("Predicted y:", predicted_y)
