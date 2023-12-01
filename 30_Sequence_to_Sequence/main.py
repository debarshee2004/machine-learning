import numpy as np


# Data preparation
def prepare_data():
    source_texts = ["hello", "world", "openai", "gpt"]
    target_texts = ["hola", "mundo", "aiabierto", "gpt"]

    source_characters = set(" ".join(source_texts))
    target_characters = set(" ".join(target_texts))

    source_char_to_index = {char: i for i, char in enumerate(source_characters)}
    target_char_to_index = {char: i for i, char in enumerate(target_characters)}

    source_index_to_char = {i: char for char, i in source_char_to_index.items()}
    target_index_to_char = {i: char for char, i in target_char_to_index.items()}

    source_vocab_size = len(source_characters)
    target_vocab_size = len(target_characters)

    max_source_seq_length = max([len(text) for text in source_texts])
    max_target_seq_length = max([len(text) for text in target_texts])

    return (
        source_texts,
        target_texts,
        source_char_to_index,
        target_char_to_index,
        source_index_to_char,
        target_index_to_char,
        source_vocab_size,
        target_vocab_size,
        max_source_seq_length,
        max_target_seq_length,
    )


# One-hot encoding
def one_hot_encode(sequence, vocab_size, char_to_index, max_seq_length):
    encoded_sequence = np.zeros((max_seq_length, vocab_size))
    for t, char in enumerate(sequence):
        encoded_sequence[t, char_to_index[char]] = 1
    return encoded_sequence


# Seq2Seq model
class Seq2Seq:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.Wxh = np.random.randn(hidden_size, input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size)
        self.Why = np.random.randn(output_size, hidden_size)

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.inputs = inputs
        self.hs = {0: h}
        self.ys = {}

        for t, x in enumerate(inputs):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.hs[t + 1] = h

        for t in range(len(inputs)):
            y = np.dot(self.Why, self.hs[t]) + self.by
            self.ys[t] = y

        return self.ys

    def backward(self, targets, learning_rate=0.01):
        dWxh, dWhh, dWhy = (
            np.zeros_like(self.Wxh),
            np.zeros_like(self.Whh),
            np.zeros_like(self.Why),
        )
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(self.hs[0])

        for t in reversed(range(len(self.inputs))):
            dy = self.ys[t] - targets[t]
            dWhy += np.dot(dy, self.hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - self.hs[t] ** 2) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, self.inputs[t].T)
            dWhh += np.dot(dhraw, self.hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # Clip to mitigate exploding gradients

        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby


# Training the Seq2Seq model
def train_seq2seq(
    source_texts,
    target_texts,
    source_char_to_index,
    target_char_to_index,
    source_index_to_char,
    target_index_to_char,
    source_vocab_size,
    target_vocab_size,
    max_source_seq_length,
    max_target_seq_length,
    hidden_size,
    epochs,
):
    model = Seq2Seq(source_vocab_size, target_vocab_size, hidden_size)

    for epoch in range(epochs):
        total_loss = 0

        for source, target in zip(source_texts, target_texts):
            source_one_hot = one_hot_encode(
                source, source_vocab_size, source_char_to_index, max_source_seq_length
            )
            target_one_hot = one_hot_encode(
                target, target_vocab_size, target_char_to_index, max_target_seq_length
            )

            model.forward(source_one_hot)
            loss = 0

            for t in range(len(target_one_hot)):
                loss += np.sum(0.5 * (model.ys[t] - target_one_hot[t]) ** 2)

            total_loss += loss

            model.backward(target_one_hot)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

    return model


# Testing the Seq2Seq model
def test_seq2seq(
    model,
    source_text,
    source_char_to_index,
    source_index_to_char,
    target_char_to_index,
    target_index_to_char,
    target_vocab_size,
    max_source_seq_length,
    max_target_seq_length,
):
    source_one_hot = one_hot_encode(
        source_text,
        len(source_char_to_index),
        source_char_to_index,
        max_source_seq_length,
    )
    predictions = model.forward(source_one_hot)

    decoded_output = ""
    for t in range(max_target_seq_length):
        predicted_index = np.argmax(predictions[t])
        if predicted_index == 0:  # Padding character
            break
        decoded_output += target_index_to_char[predicted_index]

    return decoded_output


# Main
(
    source_texts,
    target_texts,
    source_char_to_index,
    target_char_to_index,
    source_index_to_char,
    target_index_to_char,
    source_vocab_size,
    target_vocab_size,
    max_source_seq_length,
    max_target_seq_length,
) = prepare_data()

hidden_size = 128
epochs = 1000

model = train_seq2seq(
    source_texts,
    target_texts,
    source_char_to_index,
    target_char_to_index,
    source_index_to_char,
    target_index_to_char,
    source_vocab_size,
    target_vocab_size,
    max_source_seq_length,
    max_target_seq_length,
    hidden_size,
    epochs,
)

source_text = "hello"
output = test_seq2seq(
    model,
    source_text,
    source_char_to_index,
    source_index_to_char,
    target_char_to_index,
    target_index_to_char,
    target_vocab_size,
    max_source_seq_length,
    max_target_seq_length,
)

print(f"Source: {source_text}, Predicted Target: {output}")
