import numpy as np


def gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def attention(query, key, value, mask=None):
    """Scaled Dot-Product Attention."""
    d_k = query.shape[-1]
    scores = np.matmul(query, key.T) / np.sqrt(d_k)
    if mask is not None:
        scores += mask
    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, value)
    return output, weights


def softmax(x, axis=-1):
    """Softmax function."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def feedforward(x, w1, b1, w2, b2):
    """Feedforward neural network."""
    h = gelu(np.dot(x, w1) + b1)
    return np.dot(h, w2) + b2


def positional_encoding(seq_len, d_model):
    """Positional Encoding."""
    pos = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_enc = np.zeros((seq_len, d_model))
    pos_enc[:, 0::2] = np.sin(pos * div_term)
    pos_enc[:, 1::2] = np.cos(pos * div_term)
    return pos_enc


def transformer_block(x, mask, wq, wk, wv, wo, bq, bk, bv, bo, w1, b1, w2, b2):
    """Transformer block."""
    query = np.dot(x, wq) + bq
    key = np.dot(x, wk) + bk
    value = np.dot(x, wv) + bv
    output, _ = attention(query, key, value, mask)
    output += x
    output = layer_norm(output)

    # Feedforward
    ff_result = feedforward(output, w1, b1, w2, b2)
    output = ff_result + output
    output = layer_norm(output)

    return output


def layer_norm(x, epsilon=1e-5):
    """Layer Normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)


def transformer_model(input_seq, d_model, n_heads, n_layers, ff_dim):
    """Transformer Model."""
    seq_len = input_seq.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)

    # Initialize learnable parameters
    wq, bq = np.random.randn(d_model, d_model), np.zeros((1, d_model))
    wk, bk = np.random.randn(d_model, d_model), np.zeros((1, d_model))
    wv, bv = np.random.randn(d_model, d_model), np.zeros((1, d_model))
    wo, bo = np.random.randn(d_model, d_model), np.zeros((1, d_model))

    w1, b1 = np.random.randn(d_model, ff_dim), np.zeros((1, ff_dim))
    w2, b2 = np.random.randn(ff_dim, d_model), np.zeros((1, d_model))

    # Positional Encoding
    pos_enc = positional_encoding(seq_len, d_model)
    input_seq += pos_enc

    # Transformer Blocks
    for _ in range(n_layers):
        input_seq = transformer_block(
            input_seq, mask, wq, wk, wv, wo, bq, bk, bv, bo, w1, b1, w2, b2
        )

    return input_seq


# Example usage:
input_sequence = np.random.randn(
    10, 512
)  # Example input sequence with length 10 and embedding dimension 512
output_sequence = transformer_model(
    input_sequence, d_model=512, n_heads=8, n_layers=6, ff_dim=2048
)
