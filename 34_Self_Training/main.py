import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def self_training(
    X_labeled, y_labeled, X_unlabeled, model, num_iterations=10, threshold=0.8
):
    """
    Self-Training algorithm for semi-supervised learning.

    Parameters:
    - X_labeled: Labeled feature matrix
    - y_labeled: Labels for the labeled data
    - X_unlabeled: Unlabeled feature matrix
    - model: Base model (e.g., Logistic Regression)
    - num_iterations: Number of self-training iterations
    - threshold: Confidence threshold for pseudo-labeling

    Returns:
    - Updated model
    - Updated labeled data
    - Predicted labels for unlabeled data
    """

    for iteration in range(num_iterations):
        # Train the model on the labeled data
        model.fit(X_labeled, y_labeled)

        # Predict labels for the unlabeled data
        predicted_labels = model.predict(X_unlabeled)

        # Get confidence scores for the predicted labels
        confidence_scores = np.max(model.predict_proba(X_unlabeled), axis=1)

        # Filter predictions based on confidence threshold
        high_confidence_indices = confidence_scores >= threshold
        low_confidence_indices = confidence_scores < threshold

        # Add high-confidence predictions to the labeled data
        X_labeled = np.vstack([X_labeled, X_unlabeled[high_confidence_indices]])
        y_labeled = np.concatenate(
            [y_labeled, predicted_labels[high_confidence_indices]]
        )

        # Remove high-confidence predictions from the unlabeled data
        X_unlabeled = X_unlabeled[low_confidence_indices]

    return model, X_labeled, y_labeled, predicted_labels


# Example usage
# Assume X_train, y_train are the initial labeled data, and X_unlabeled is the unlabeled data
# Also, assume logistic regression as the base model

# Initialize the base model
base_model = LogisticRegression()

# Perform self-training
final_model, X_final_labeled, y_final_labeled, predicted_labels = self_training(
    X_train, y_train, X_unlabeled, base_model, num_iterations=5, threshold=0.8
)

# Evaluate the final model on the labeled data
final_accuracy = accuracy_score(y_final_labeled, final_model.predict(X_final_labeled))
print(f"Final Model Accuracy: {final_accuracy}")
