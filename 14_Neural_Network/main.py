import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases for the hidden and output layers
        self.weights_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid activation function
        return x * (1 - x)

    def forward(self, inputs):
        # Forward propagation
        self.hidden_layer_input = np.dot(inputs, self.weights_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = (
            np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        )
        self.output_layer_output = self.sigmoid(self.output_layer_input)

        return self.output_layer_output

    def backward(self, inputs, targets, learning_rate):
        # Backward propagation
        output_error = targets - self.output_layer_output
        output_delta = output_error * self.sigmoid_derivative(self.output_layer_output)

        hidden_error = output_delta.dot(self.weights_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_output += (
            self.hidden_layer_output.T.dot(output_delta) * learning_rate
        )
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_hidden += inputs.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        # Train the neural network
        for epoch in range(epochs):
            # Forward and backward pass for each training example
            for input_data, target in zip(inputs, targets):
                input_data = np.reshape(input_data, (1, -1))
                target = np.reshape(target, (1, -1))
                self.forward(input_data)
                self.backward(input_data, target, learning_rate)

            # Print the mean squared error for every 100 epochs
            if epoch % 100 == 0:
                mse = np.mean(np.square(targets - self.forward(inputs)))
                print(f"Epoch {epoch}, Mean Squared Error: {mse}")


# Example usage
input_size = 2
hidden_size = 3
output_size = 1

# Training data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Create and train the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(inputs, targets, epochs=1000, learning_rate=0.1)

# Test the trained network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.forward(test_data)
print("Predictions:")
print(predictions)
