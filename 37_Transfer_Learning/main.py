import numpy as np


def initialize_model(input_size, num_classes):
    # Initialize a simple neural network model (for demonstration purposes)
    model = {
        "weights_input_hidden": np.random.randn(input_size, 64),
        "bias_hidden": np.zeros((1, 64)),
        "weights_hidden_output": np.random.randn(64, num_classes),
        "bias_output": np.zeros((1, num_classes)),
    }
    return model


def forward_pass(X, model):
    # Forward pass through the neural network
    hidden_layer = np.dot(X, model["weights_input_hidden"]) + model["bias_hidden"]
    hidden_layer_activation = np.maximum(0, hidden_layer)  # ReLU activation
    output_layer = (
        np.dot(hidden_layer_activation, model["weights_hidden_output"])
        + model["bias_output"]
    )
    return hidden_layer_activation, output_layer


def compute_loss(predictions, targets):
    # Mean Squared Error loss (for demonstration purposes)
    loss = np.mean((predictions - targets) ** 2)
    return loss


def fine_tune_model(model, new_data, new_targets, learning_rate=0.01, num_epochs=100):
    for epoch in range(num_epochs):
        for i in range(len(new_data)):
            # Forward pass
            hidden_layer_activation, predictions = forward_pass(new_data[i], model)

            # Compute loss
            loss = compute_loss(predictions, new_targets[i])

            # Backward pass (gradient descent)
            output_error = predictions - new_targets[i]
            hidden_error = np.dot(output_error, model["weights_hidden_output"].T)
            hidden_error[hidden_layer_activation <= 0] = 0  # ReLU derivative

            # Update weights and biases
            model["weights_hidden_output"] -= learning_rate * np.dot(
                hidden_layer_activation.T, output_error
            )
            model["bias_output"] -= learning_rate * np.sum(
                output_error, axis=0, keepdims=True
            )
            model["weights_input_hidden"] -= learning_rate * np.dot(
                new_data[i].T, hidden_error
            )
            model["bias_hidden"] -= learning_rate * np.sum(
                hidden_error, axis=0, keepdims=True
            )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    return model


# Example usage
input_size = 10  # Adjust based on your input features
num_classes = 1  # Adjust based on your output classes
pretrained_model = initialize_model(input_size, num_classes)

# Generate new data for fine-tuning (replace with your own data)
new_data = np.random.randn(100, input_size)
new_targets = np.random.randn(100, num_classes)

# Fine-tune the model
fine_tuned_model = fine_tune_model(pretrained_model, new_data, new_targets)
