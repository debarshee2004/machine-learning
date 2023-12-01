import numpy as np


class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, current_depth=0):
        if current_depth >= self.max_depth or len(X) <= 1:
            return {"isolation_score": 2**-current_depth}

        num_features = X.shape[1]
        split_feature = np.random.randint(num_features)
        split_value = np.random.uniform(
            X[:, split_feature].min(), X[:, split_feature].max()
        )

        left_mask = X[:, split_feature] < split_value
        right_mask = ~left_mask

        left_subtree = self.fit(X[left_mask], current_depth + 1)
        right_subtree = self.fit(X[right_mask], current_depth + 1)

        return {
            "split_feature": split_feature,
            "split_value": split_value,
            "left_subtree": left_subtree,
            "right_subtree": right_subtree,
        }


class IsolationForest:
    def __init__(self, n_trees, max_depth):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X):
        for _ in range(self.n_trees):
            tree = IsolationTree(max_depth=self.max_depth)
            self.trees.append(tree.fit(X))

    def anomaly_score(self, x):
        scores = [self.path_length(x, tree) for tree in self.trees]
        average_path_length = np.mean(scores)
        return 2 ** (-average_path_length / self.c_avg(len(x)))

    def path_length(self, x, tree, current_depth=0):
        if "split_feature" not in tree:
            return current_depth
        split_feature = tree["split_feature"]
        split_value = tree["split_value"]

        if x[split_feature] < split_value:
            return self.path_length(x, tree["left_subtree"], current_depth + 1)
        else:
            return self.path_length(x, tree["right_subtree"], current_depth + 1)

    def c_avg(self, size):
        if size <= 2:
            return 1
        return 2 * (np.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size


# Example usage:
# Create a random dataset for demonstration
np.random.seed(42)
data = np.random.rand(100, 2)

# Train the Isolation Forest
isolation_forest = IsolationForest(n_trees=100, max_depth=10)
isolation_forest.fit(data)

# Calculate anomaly scores for each data point
anomaly_scores = np.array([isolation_forest.anomaly_score(point) for point in data])

# Set a threshold to classify anomalies
threshold = 0.3
anomalies = data[anomaly_scores > threshold]

print("Anomalies:")
print(anomalies)
