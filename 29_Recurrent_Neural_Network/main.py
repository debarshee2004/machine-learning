import numpy as np


# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Function to initialize weights and biases
def initialize_parameters(input_size, hidden_size, output_size):
    Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
    Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
    bh = np.zeros((hidden_size, 1))  # hidden bias
    by = np.zeros((output_size, 1))  # output bias

    parameters = {"Wxh": Wxh, "Whh": Whh, "Why": Why, "bh": bh, "by": by}
    return parameters


# Function to perform forward propagation
def forward_propagation(inputs, parameters):
    Wxh, Whh, Why = parameters["Wxh"], parameters["Whh"], parameters["Why"]
    bh, by = parameters["bh"], parameters["by"]

    h_prev = np.zeros((Whh.shape[0], 1))  # initialize hidden state

    cache = {
        "h_prev": h_prev,
        "inputs": inputs,
        "Wxh": Wxh,
        "Whh": Whh,
        "Why": Why,
        "bh": bh,
        "by": by,
    }

    # Lists to store intermediate values for backpropagation
    xs, hs, ys, ps = [], [], [], []

    for t in range(inputs.shape[1]):
        xs.append(inputs[:, t].reshape(-1, 1))
        hs.append(np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, h_prev) + bh))
        ys.append(np.dot(Why, hs[t]) + by)
        ps.append(sigmoid(ys[t]))
        h_prev = hs[t]

    cache["xs"], cache["hs"], cache["ys"], cache["ps"] = xs, hs, ys, ps

    return ps, cache


# Function to perform backward propagation and compute gradients
def backward_propagation(ps, targets, cache):
    Wxh, Whh, Why = cache["Wxh"], cache["Whh"], cache["Why"]
    bh, by = cache["bh"], cache["by"]
    xs, hs, ys = cache["xs"], cache["hs"], cache["ys"]

    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(xs))):
        dy = ps[t] - targets[:, t]
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dh_next
        dh_raw = (1 - hs[t] ** 2) * dh  # backprop through tanh

        dWxh += np.dot(dh_raw, xs[t].T)
        dWhh += np.dot(dh_raw, hs[t - 1].T)
        dbh += dh_raw
        dh_next = np.dot(Whh.T, dh_raw)

    gradients = {"dWxh": dWxh, "dWhh": dWhh, "dWhy": dWhy, "dbh": dbh, "dby": dby}

    return gradients


# Function to update parameters using gradients and learning rate
def update_parameters(parameters, gradients, learning_rate=0.01):
    for param_name in parameters:
        parameters[param_name] -= learning_rate * gradients["d" + param_name]


# Function to train the RNN
def train_rnn(inputs, targets, hidden_size, epochs):
    input_size = inputs.shape[0]
    output_size = targets.shape[0]

    parameters = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        ps, cache = forward_propagation(inputs, parameters)
        gradients = backward_propagation(ps, targets, cache)
        update_parameters(parameters, gradients)

        if epoch % 100 == 0:
            loss = (
                -np.sum(targets * np.log(ps) + (1 - targets) * np.log(1 - ps))
                / targets.shape[1]
            )
            print(f"Epoch {epoch}, Loss: {loss:.4f}")


# Dummy data for demonstration
seq_length = 5
input_size = 3
hidden_size = 4
output_size = 1

inputs = np.random.randn(input_size, seq_length)
targets = np.random.randint(2, size=(output_size, seq_length))

# Training the RNN
train_rnn(inputs, targets, hidden_size, epochs=1000)
