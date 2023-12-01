import numpy as np


class StackingEnsemble:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit_base_models(self, X, y):
        for model in self.base_models:
            model.fit(X, y)

    def predict_base_models(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.base_models])
        return predictions

    def fit_meta_model(self, X, y):
        self.meta_model.fit(X, y)

    def predict_meta_model(self, X):
        return self.meta_model.predict(X)

    def fit(self, X, y):
        self.fit_base_models(X, y)
        base_model_predictions = self.predict_base_models(X)
        self.fit_meta_model(base_model_predictions, y)

    def predict(self, X):
        base_model_predictions = self.predict_base_models(X)
        return self.predict_meta_model(base_model_predictions)


# Example usage
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 5)
y = (X.sum(axis=1) > 2.5).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define base models
base_model1 = DecisionTreeClassifier(random_state=42)
base_model2 = RandomForestClassifier(random_state=42)
base_model3 = LogisticRegression(random_state=42)

# Define meta-model
meta_model = LogisticRegression(random_state=42)

# Create StackingEnsemble
ensemble = StackingEnsemble(
    base_models=[base_model1, base_model2, base_model3], meta_model=meta_model
)

# Train the ensemble
ensemble.fit(X_train, y_train)

# Make predictions on the test set
predictions = ensemble.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Ensemble Accuracy: {accuracy}")
