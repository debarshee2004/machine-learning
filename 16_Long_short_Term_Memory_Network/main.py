import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def lstm_cell(xt, a_prev, c_prev, parameters):
    """
    Implement a single LSTM cell.

    Arguments:
    - xt: input data at time step t
    - a_prev: hidden state at time step t-1
    - c_prev: cell state at time step t-1
    - parameters: dictionary containing parameters (Wf, bf, Wi, bi, Wo, bo, Wc, bc)

    Returns:
    - a_next: next hidden state
    - c_next: next cell state
    - yt_pred: prediction at time step t
    """
    Wf, bf, Wi, bi, Wo, bo, Wc, bc = parameters

    # Concatenate input and previous hidden state
    concat = np.concatenate((a_prev, xt), axis=1)

    # Forget gate
    ft = sigmoid(np.dot(concat, Wf) + bf)

    # Input gate
    it = sigmoid(np.dot(concat, Wi) + bi)

    # Output gate
    ot = sigmoid(np.dot(concat, Wo) + bo)

    # Cell state candidates
    cct = tanh(np.dot(concat, Wc) + bc)

    # Update cell state and hidden state
    c_next = ft * c_prev + it * cct
    a_next = ot * tanh(c_next)

    # Prediction at time step t
    yt_pred = a_next

    return a_next, c_next, yt_pred


def lstm_forward(x, a0, parameters):
    """
    Implement forward propagation for an LSTM.

    Arguments:
    - x: input data for all time steps
    - a0: initial hidden state
    - parameters: dictionary containing parameters (Wf, bf, Wi, bi, Wo, bo, Wc, bc)

    Returns:
    - a: hidden states for all time steps
    - y_pred: predictions for all time steps
    - caches: tuple containing values needed for backpropagation
    """
    caches = []
    n_x, m, T_x = x.shape
    n_a = a0.shape[0]

    a = np.zeros((n_a, m, T_x))
    c = np.zeros_like(a)
    y_pred = np.zeros_like(a)

    a_next = a0
    c_next = np.zeros_like(a_next)

    for t in range(T_x):
        a_next, c_next, yt_pred = lstm_cell(x[:, :, t], a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y_pred[:, :, t] = yt_pred

        caches.append((a_next, c_next, parameters))

    caches = (caches, x)

    return a, y_pred, caches


# Example usage:
np.random.seed(1)
n_x = 3  # Input size
m = 5  # Number of examples
T_x = 4  # Number of time steps

# Initialize parameters
Wf = np.random.randn(n_x + m, n_x + m)
bf = np.zeros((1, n_x + m))
Wi = np.random.randn(n_x + m, n_x + m)
bi = np.zeros((1, n_x + m))
Wo = np.random.randn(n_x + m, n_x + m)
bo = np.zeros((1, n_x + m))
Wc = np.random.randn(n_x + m, n_x + m)
bc = np.zeros((1, n_x + m))

parameters = (Wf, bf, Wi, bi, Wo, bo, Wc, bc)

# Initialize input data and initial hidden state
x = np.random.randn(n_x, m, T_x)
a0 = np.zeros((n_x + m, m))

# Forward pass
a, y_pred, caches = lstm_forward(x, a0, parameters)

# Print the results
print("Hidden states:")
print(a)
print("\nPredictions:")
print(y_pred)
