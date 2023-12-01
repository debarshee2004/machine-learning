import numpy as np


def compute_pairwise_squared_distances(X):
    """
    Compute pairwise squared Euclidean distances between points in X.

    Parameters:
    - X: NumPy array, shape (n_samples, n_features)

    Returns:
    - distances: NumPy array, shape (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            distances[i, j] = np.sum((X[i] - X[j]) ** 2)

    return distances


def compute_conditional_probabilities(distances, perplexity=30.0):
    """
    Compute conditional probabilities using the Student's t-distribution.

    Parameters:
    - distances: NumPy array, shape (n_samples, n_samples)
    - perplexity: float, target perplexity for the distribution

    Returns:
    - P: NumPy array, shape (n_samples, n_samples)
    """
    n_samples = distances.shape[0]
    P = np.zeros((n_samples, n_samples))
    beta = np.ones((n_samples, 1))
    target_entropy = np.log(perplexity)

    for i in range(n_samples):
        # Binary search to find the appropriate sigma (beta in the original paper)
        lower_bound = -np.inf
        upper_bound = np.inf
        tol = 1e-5
        max_iter = 1000
        sigma = 1.0

        for _ in range(max_iter):
            exp_distances = np.exp(-distances[i] * beta[i])
            sum_exp_distances = np.sum(exp_distances)
            entropy = np.sum(distances[i] * exp_distances) / sum_exp_distances + np.log(
                sum_exp_distances
            )

            diff = entropy - target_entropy

            if np.abs(diff) < tol:
                break

            if diff > 0:
                upper_bound = beta[i]
                if np.isinf(upper_bound):
                    beta[i] /= 2.0
                else:
                    beta[i] = (beta[i] + lower_bound) / 2.0
            else:
                lower_bound = beta[i]
                if np.isinf(upper_bound):
                    beta[i] *= 2.0
                else:
                    beta[i] = (beta[i] + upper_bound) / 2.0

        # Compute conditional probabilities
        P[i] = exp_distances / sum_exp_distances

    # Ensure symmetry
    P = (P + P.T) / (2.0 * n_samples)

    return P


def compute_grad(y, P, Q):
    """
    Compute gradient of the cost function.

    Parameters:
    - y: NumPy array, shape (n_samples, n_components)
    - P: NumPy array, shape (n_samples, n_samples)
    - Q: NumPy array, shape (n_samples, n_samples)

    Returns:
    - grad: NumPy array, shape (n_samples, n_components)
    """
    n_samples, n_components = y.shape

    # Compute difference matrix
    Dy = np.tile(y[:, np.newaxis, :], (1, n_samples, 1)) - np.tile(
        y[np.newaxis, :, :], (n_samples, 1, 1)
    )
    square_distances = np.sum(Dy**2, axis=2)

    # Compute gradients
    factor = P - Q
    grad = np.zeros((n_samples, n_components))

    for i in range(n_samples):
        grad[i] = np.sum(
            np.tile(factor[i, :, np.newaxis], (1, n_components)) * Dy[i], axis=0
        )

    return grad


def t_sne(X, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE).

    Parameters:
    - X: NumPy array, shape (n_samples, n_features)
    - n_components: int, number of dimensions in the embedded space
    - perplexity: float, target perplexity for the distribution
    - learning_rate: float, learning rate for the optimization
    - n_iter: int, number of iterations

    Returns:
    - Y: NumPy array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape

    # Initialize low-dimensional representation randomly
    Y = np.random.randn(n_samples, n_components)

    # Compute pairwise squared distances in the original space
    distances = compute_pairwise_squared_distances(X)

    # Initialize P matrix
    P = compute_conditional_probabilities(distances, perplexity)

    # Initialize Q matrix
    Q = np.zeros((n_samples, n_samples))

    # Perform gradient descent
    for iteration in range(n_iter):
        # Compute pairwise squared distances in the embedded space
        embedding_distances = compute_pairwise_squared_distances(Y)

        # Compute Q matrix
        Q = 1.0 / (1.0 + embedding_distances)
        np.fill_diagonal(Q, 0.0)
        Q /= np.sum(Q)

        # Compute gradient of the cost function
        grad = compute_grad(Y, P, Q)

        # Update embedding
        Y -= learning_rate * grad

        # Print progress every 100 iterations
        if iteration % 100 == 0:
            cost = np.sum(P * np.log(P / Q))
            print(f"Iteration {iteration}/{n_iter}, Cost: {cost}")

    return Y


# Example usage:
# Assuming you have your data in a NumPy array X with shape (n_samples, n_features)
# Y = t_sne(X, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000)
