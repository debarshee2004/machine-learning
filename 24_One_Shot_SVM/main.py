import numpy as np
from numpy import linalg as LA


class OneClassSVM:
    def __init__(self, nu=0.01, kernel="rbf", gamma=0.1):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.support_vectors = None
        self.intercept = None

    def fit(self, X):
        # Choose the kernel function
        if self.kernel == "rbf":
            kernel_func = self._rbf_kernel
        else:
            raise ValueError("Unsupported kernel function")

        # Fit the model using the One-Class SVM optimization problem
        n_samples, n_features = X.shape
        gram_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                gram_matrix[i, j] = kernel_func(X[i], X[j])

        # Solve the optimization problem using the SMO algorithm
        P = np.outer(np.ones(n_samples), np.ones(n_samples))
        P = P * gram_matrix
        q = -np.ones(n_samples)
        G = -np.eye(n_samples)
        h = np.zeros(n_samples)
        A = np.ones(n_samples)
        b = np.ones(1) * self.nu * n_samples

        alpha = self._solve_qp(P, q, G, h, A, b)

        # Identify support vectors
        support_vector_indices = np.where(alpha > 1e-5)[0]
        self.support_vectors = X[support_vector_indices]

        # Compute intercept
        sum_alpha_y = np.sum(alpha[support_vector_indices])
        sum_kernel = np.sum(
            kernel_func(self.support_vectors, self.support_vectors), axis=1
        )
        self.intercept = 1 / len(support_vector_indices) * (sum_alpha_y - sum_kernel)

    def predict(self, X):
        # Make predictions for new data points
        if self.support_vectors is None or self.intercept is None:
            raise ValueError(
                "The model has not been trained yet. Please call the fit method first."
            )

        kernel_values = self._rbf_kernel(X, self.support_vectors)
        decision_function = np.dot(kernel_values, self.intercept) - 1

        # Predictions: -1 for outliers, 1 for inliers
        predictions = np.sign(decision_function)

        return predictions

    def _solve_qp(self, P, q, G, h, A, b):
        # Solve the quadratic programming problem
        from cvxopt import matrix, solvers

        solvers.options["show_progress"] = False
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A, (1, len(A)))
        b = matrix(b)
        solution = solvers.qp(P, q, G, h, A, b)
        return np.array(solution["x"]).flatten()

    def _rbf_kernel(self, x, y):
        # Radial basis function (RBF) kernel
        return np.exp(-self.gamma * LA.norm(x - y) ** 2)


# Example usage
np.random.seed(42)
normal_data = np.random.normal(0, 1, (100, 2))
anomalous_data = np.random.normal(5, 1, (10, 2))

# Combine normal and anomalous data
data = np.vstack((normal_data, anomalous_data))

# Shuffle the data
np.random.shuffle(data)

# Instantiate and fit the OneClassSVM model
ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
ocsvm.fit(data)

# Make predictions
predictions = ocsvm.predict(data)

# Print the predictions
print(predictions)
