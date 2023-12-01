import numpy as np
from collections import defaultdict


def preprocess_text(text):
    # Tokenize the text into words
    words = text.lower().split()
    return words


def create_one_hot_vector(word_index, vocab_size):
    one_hot_vector = np.zeros(vocab_size)
    one_hot_vector[word_index] = 1
    return one_hot_vector


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)


def initialize_parameters(vocab_size, embedding_size):
    # Initialize the word vectors and context vectors
    word_vectors = np.random.rand(vocab_size, embedding_size)
    context_vectors = np.random.rand(embedding_size, vocab_size)

    return word_vectors, context_vectors


def skipgram_model_training(corpus, window_size, embedding_size, learning_rate, epochs):
    # Create vocabulary and mapping of words to indices
    vocab = list(set(corpus))
    vocab_size = len(vocab)
    word_to_index = {word: index for index, word in enumerate(vocab)}
    index_to_word = {index: word for index, word in enumerate(vocab)}

    # Initialize parameters
    word_vectors, context_vectors = initialize_parameters(vocab_size, embedding_size)

    # Training loop
    for epoch in range(epochs):
        for center_word_index, center_word in enumerate(corpus):
            context_indices = list(
                range(
                    max(0, center_word_index - window_size),
                    min(len(corpus), center_word_index + window_size + 1),
                )
            )
            context_indices.remove(center_word_index)

            for context_index in context_indices:
                # Forward pass
                center_word_vector = word_vectors[word_to_index[center_word]]
                context_word_index = word_to_index[corpus[context_index]]
                predicted_vector = np.dot(
                    center_word_vector, context_vectors[:, context_word_index]
                )
                predicted_probs = softmax(predicted_vector)

                # Loss computation (cross-entropy)
                loss = -np.log(predicted_probs[context_word_index])

                # Backward pass (gradient descent)
                grad_predicted = predicted_probs
                grad_predicted[context_word_index] -= 1
                grad_context = np.outer(center_word_vector, grad_predicted)

                # Update parameters
                word_vectors[word_to_index[center_word]] -= (
                    learning_rate * grad_context[:, context_word_index]
                )
                context_vectors[:, context_word_index] -= learning_rate * grad_predicted

    return word_vectors, word_to_index, index_to_word


# Example usage
text = "natural language processing is a field of artificial intelligence"
corpus = preprocess_text(text)
window_size = 2
embedding_size = 10
learning_rate = 0.01
epochs = 1000

word_vectors, word_to_index, index_to_word = skipgram_model_training(
    corpus, window_size, embedding_size, learning_rate, epochs
)

# Print word vectors
for word, index in word_to_index.items():
    print(f"{word}: {word_vectors[index]}")
