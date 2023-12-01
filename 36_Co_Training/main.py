import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def co_training(X_labeled, y_labeled, X_unlabeled, clf1, clf2, num_rounds=10):
    for _ in range(num_rounds):
        # Train and predict on view 1
        clf1.fit(X_labeled, y_labeled)
        pred1 = clf1.predict(X_unlabeled)

        # Train and predict on view 2
        clf2.fit(X_labeled, y_labeled)
        pred2 = clf2.predict(X_unlabeled)

        # Select the most confident predictions from each view
        confident_indices1 = np.argsort(
            np.max(clf1.predict_proba(X_unlabeled), axis=1)
        )[-len(X_labeled) :]
        confident_indices2 = np.argsort(
            np.max(clf2.predict_proba(X_unlabeled), axis=1)
        )[-len(X_labeled) :]

        # Update the labeled dataset with the agreed-upon predictions
        agreed_indices = np.intersect1d(confident_indices1, confident_indices2)
        X_labeled = np.vstack((X_labeled, X_unlabeled[agreed_indices]))
        y_labeled = np.concatenate((y_labeled, pred1[agreed_indices]))

        # Remove the agreed-upon instances from the unlabeled dataset
        X_unlabeled = np.delete(X_unlabeled, agreed_indices, axis=0)

    return clf1, clf2


# Example usage:
# Assume you have X_labeled, y_labeled, and X_unlabeled as your labeled and unlabeled datasets
# Assume clf1 and clf2 are your base classifiers (e.g., DecisionTreeClassifier)

# Split the data into labeled and unlabeled sets
num_labeled = 100  # adjust this based on the amount of labeled data you have
X_labeled = X[:num_labeled]
y_labeled = y[:num_labeled]
X_unlabeled = X[num_labeled:]

# Create two classifiers
clf1 = DecisionTreeClassifier()
clf2 = DecisionTreeClassifier()

# Apply Co-Training
clf1, clf2 = co_training(X_labeled, y_labeled, X_unlabeled, clf1, clf2)

# Evaluate the performance on a test set or use the models for predictions
