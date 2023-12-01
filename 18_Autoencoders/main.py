import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize_parameters(input_size, hidden_size):
    np.random.seed(42)
    weights_encoder = np.random.randn(hidden_size, input_size)
    bias_encoder = np.zeros((hidden_size, 1))
    weights_decoder = np.random.randn(input_size, hidden_size)
    bias_decoder = np.zeros((input_size, 1))
    return weights_encoder, bias_encoder, weights_decoder, bias_decoder


def forward_propagation(
    X, weights_encoder, bias_encoder, weights_decoder, bias_decoder
):
    encoded = sigmoid(np.dot(weights_encoder, X) + bias_encoder)
    decoded = sigmoid(np.dot(weights_decoder, encoded) + bias_decoder)
    return encoded, decoded


def backward_propagation(X, encoded, decoded, weights_encoder, weights_decoder):
    error = X - decoded
    delta_decoder = error * sigmoid_derivative(decoded)
    delta_encoder = np.dot(weights_decoder.T, delta_decoder) * sigmoid_derivative(
        encoded
    )
    return delta_encoder, delta_decoder


def update_parameters(
    X,
    encoded,
    delta_encoder,
    delta_decoder,
    weights_encoder,
    bias_encoder,
    weights_decoder,
    bias_decoder,
    learning_rate,
):
    weights_encoder += learning_rate * np.dot(delta_encoder, X.T)
    bias_encoder += learning_rate * np.sum(delta_encoder, axis=1, keepdims=True)
    weights_decoder += learning_rate * np.dot(delta_decoder, encoded.T)
    bias_decoder += learning_rate * np.sum(delta_decoder, axis=1, keepdims=True)
    return weights_encoder, bias_encoder, weights_decoder, bias_decoder


def train_autoencoder(X, hidden_size, epochs, learning_rate):
    input_size = X.shape[0]
    (
        weights_encoder,
        bias_encoder,
        weights_decoder,
        bias_decoder,
    ) = initialize_parameters(input_size, hidden_size)

    for epoch in range(epochs):
        encoded, decoded = forward_propagation(
            X, weights_encoder, bias_encoder, weights_decoder, bias_decoder
        )
        delta_encoder, delta_decoder = backward_propagation(
            X, encoded, decoded, weights_encoder, weights_decoder
        )
        (
            weights_encoder,
            bias_encoder,
            weights_decoder,
            bias_decoder,
        ) = update_parameters(
            X,
            encoded,
            delta_encoder,
            delta_decoder,
            weights_encoder,
            bias_encoder,
            weights_decoder,
            bias_decoder,
            learning_rate,
        )

        if epoch % 100 == 0:
            loss = np.mean(np.abs(X - decoded))
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights_encoder, bias_encoder, weights_decoder, bias_decoder


# Example usage:
if __name__ == "__main__":
    # Generate some random data for demonstration
    np.random.seed(42)
    data = np.random.rand(5, 100)

    # Define hyperparameters
    hidden_size = 3
    epochs = 1000
    learning_rate = 0.01

    # Train the autoencoder
    (
        trained_weights_encoder,
        trained_bias_encoder,
        trained_weights_decoder,
        trained_bias_decoder,
    ) = train_autoencoder(data, hidden_size, epochs, learning_rate)

    # Test the autoencoder
    _, reconstructed_data = forward_propagation(
        data,
        trained_weights_encoder,
        trained_bias_encoder,
        trained_weights_decoder,
        trained_bias_decoder,
    )

    # Print original and reconstructed data
    print("Original Data:")
    print(data)
    print("Reconstructed Data:")
    print(reconstructed_data)
