import numpy as np


class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_iter=100):
        self.C = C  # Regularization parameter
        self.tol = tol  # Tolerance for stopping criterion
        self.max_iter = max_iter  # Maximum number of iterations
        self.b = 0  # Bias term
        self.alphas = None  # Lagrange multipliers
        self.X = None  # Input data
        self.y = None  # Labels
        self.m = None  # Number of data points
        self.n = None  # Number of features

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.m, self.n = X.shape
        self.alphas = np.zeros(self.m)
        self.b = 0

        # SMO (Sequential Minimal Optimization)  algorithm
        num_changed_alphas = 0
        iteration = 0
        while (num_changed_alphas > 0 or iteration == 0) and (
            iteration < self.max_iter
        ):
            num_changed_alphas = 0
            for i in range(self.m):
                E_i = self.decision_function(X[i]) - y[i]
                if (y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (
                    y[i] * E_i > self.tol and self.alphas[i] > 0
                ):
                    j = self.select_second_alpha(i)
                    E_j = self.decision_function(X[j]) - y[j]

                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    eta = 2 * X[i].dot(X[j]) - X[i].dot(X[i]) - X[j].dot(X[j])
                    if eta >= 0:
                        continue

                    self.alphas[j] -= y[j] * (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)

                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])

                    b1 = (
                        self.b
                        - E_i
                        - y[i] * (self.alphas[i] - alpha_i_old) * X[i].dot(X[i])
                        - y[j] * (self.alphas[j] - alpha_j_old) * X[i].dot(X[j])
                    )
                    b2 = (
                        self.b
                        - E_j
                        - y[i] * (self.alphas[i] - alpha_i_old) * X[i].dot(X[j])
                        - y[j] * (self.alphas[j] - alpha_j_old) * X[j].dot(X[j])
                    )

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                iteration += 1
            else:
                iteration = 0

    def decision_function(self, X):
        return np.dot(self.alphas * self.y, self.kernel_function(X, self.X)) - self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def select_second_alpha(self, i):
        j = i
        while j == i:
            j = np.random.randint(self.m)
        return j

    @staticmethod
    def kernel_function(x1, x2):
        # Linear kernel function
        return np.dot(x1, x2)


# Example usage:
# Assuming you have training data X_train and labels y_train
# and test data X_test.

# Train SVM
svm = SVM(C=1.0, tol=1e-3, max_iter=100)
svm.fit(X_train, y_train)

# Make predictions
predictions = svm.predict(X_test)
