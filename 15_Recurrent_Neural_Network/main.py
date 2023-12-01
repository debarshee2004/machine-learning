import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W_xh = np.random.randn(hidden_size, input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size)
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

        # Store intermediate values for backpropagation
        self.x, self.h, self.y = {}, {}, {}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        T = len(inputs)
        self.h[-1] = np.zeros((self.hidden_size, 1))

        for t in range(T):
            self.x[t] = inputs[t].reshape(-1, 1)
            self.h[t] = np.tanh(
                np.dot(self.W_xh, self.x[t])
                + np.dot(self.W_hh, self.h[t - 1])
                + self.b_h
            )
            self.y[t] = self.sigmoid(np.dot(self.W_hy, self.h[t]) + self.b_y)

        return self.y

    def backward(self, targets, learning_rate=0.01):
        T = len(targets)
        dW_xh, dW_hh, dW_hy = (
            np.zeros_like(self.W_xh),
            np.zeros_like(self.W_hh),
            np.zeros_like(self.W_hy),
        )
        db_h, db_y = np.zeros_like(self.b_h), np.zeros_like(self.b_y)
        dh_next = np.zeros_like(self.h[0])

        for t in reversed(range(T)):
            dy = self.y[t] - targets[t].reshape(-1, 1)
            dW_hy += np.dot(dy, self.h[t].T)
            db_y += dy

            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - self.h[t] ** 2) * dh  # Derivative of tanh
            db_h += dh_raw
            dW_xh += np.dot(dh_raw, self.x[t].T)
            dW_hh += np.dot(dh_raw, self.h[t - 1].T)
            dh_next = np.dot(self.W_hh.T, dh_raw)

        # Update weights and biases
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

    def train(self, inputs, targets, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(inputs)):
                self.forward(inputs[i])
                self.backward(targets[i], learning_rate)
                total_loss += np.sum((self.y[i] - targets[i]) ** 2)
            avg_loss = total_loss / len(inputs)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


# Example usage:
input_size = 3
hidden_size = 4
output_size = 1

rnn = SimpleRNN(input_size, hidden_size, output_size)

# Generate some dummy data for training
inputs = [np.array([[0.1], [0.2], [0.3]]), np.array([[0.4], [0.5], [0.6]])]
targets = [np.array([[0.4]]), np.array([[0.7]])]

rnn.train(inputs, targets, epochs=100, learning_rate=0.01)
