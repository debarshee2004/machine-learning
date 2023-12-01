import numpy as np


def pca(X, num_components):
    # Standardize the data (subtract mean and divide by standard deviation)
    X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Calculate the covariance matrix
    covariance_matrix = np.cov(X_standardized, rowvar=False)

    # Calculate eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'num_components' eigenvectors
    top_eigenvectors = eigenvectors[:, :num_components]

    # Project the standardized data onto the top eigenvectors to obtain the principal components
    principal_components = np.dot(X_standardized, top_eigenvectors)

    return principal_components, top_eigenvectors


# Example usage:
# Assuming you have a dataset X with samples in rows and features in columns
# num_components is the desired number of principal components
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

num_components = 2
principal_components, top_eigenvectors = pca(X, num_components)

# principal_components now contains the reduced-dimensional representation of the data
# top_eigenvectors contains the top eigenvectors corresponding to the principal components
