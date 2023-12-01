import numpy as np


def relu(x):
    return np.maximum(0, x)


def convolution2d(image, kernel):
    # Assuming 'image' and 'kernel' are 2D arrays
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    o_h, o_w = i_h - k_h + 1, i_w - k_w + 1

    output = np.zeros((o_h, o_w))

    for i in range(o_h):
        for j in range(o_w):
            output[i, j] = np.sum(image[i : i + k_h, j : j + k_w] * kernel)

    return output


def maxpool2d(image, pool_size=(2, 2)):
    p_h, p_w = pool_size
    o_h, o_w = image.shape[0] // p_h, image.shape[1] // p_w

    output = np.zeros((o_h, o_w))

    for i in range(o_h):
        for j in range(o_w):
            output[i, j] = np.max(
                image[i * p_h : (i + 1) * p_h, j * p_w : (j + 1) * p_w]
            )

    return output


# Simple Convolutional Neural Network
class SimpleCNN:
    def __init__(self):
        # Convolutional layer weights and bias
        self.conv_weights = np.random.randn(3, 3)  # Example 3x3 kernel
        self.conv_bias = np.zeros(1)

        # Fully connected layer weights and bias
        self.fc_weights = np.random.randn(
            9, 10
        )  # Example weights for 9 input features to 10 output neurons
        self.fc_bias = np.zeros(10)

    def forward(self, input_data):
        # Convolutional layer
        conv_output = convolution2d(input_data, self.conv_weights) + self.conv_bias
        conv_output = relu(conv_output)

        # Maxpooling layer
        pooled_output = maxpool2d(conv_output)

        # Flatten the output for the fully connected layer
        flattened_output = pooled_output.flatten()

        # Fully connected layer
        fc_output = np.dot(flattened_output, self.fc_weights) + self.fc_bias
        fc_output = relu(fc_output)

        return fc_output


# Example usage
if __name__ == "__main__":
    # Input image (random data for demonstration)
    input_image = np.random.randn(28, 28)  # Example 28x28 image

    # Instantiate the CNN
    cnn = SimpleCNN()

    # Forward pass
    output = cnn.forward(input_image)

    print("Output shape:", output.shape)
    print("Output values:", output)
