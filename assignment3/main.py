import numpy as np
from torchvision.datasets import MNIST

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)

# Load data
train_X, train_Y = download_mnist(is_train=True)
test_X, test_Y = download_mnist(is_train=False)

# Normalize data
train_X = train_X / 255.0
test_X = test_X / 255.0

# One-hot encoding for labels
def get_one_hot_encoding(y, size=10):
    return np.eye(size)[y]

train_y_hot = get_one_hot_encoding(train_Y)
test_y_hot = get_one_hot_encoding(test_Y)

# Network parameters
input_neurons = 784
hidden_neurons = 100
output_neurons = 10
learning_rate = 0.01
epochs = 100
batch_size = 100
dropout_rate = 0.5

np.random.seed(42)
W1 = np.random.randn(input_neurons, hidden_neurons) * 0.01
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, output_neurons) * 0.01
b2 = np.zeros((1, output_neurons))

# Activation functions

#hidden
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#output
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability fix
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

#derivatives
def sigmoid_derivative(a):
    return np.clip(a * (1 - a), 1e-7, 1 - 1e-7)  # avoid overflow

#forward propagation with dropout
def forward_propagation(X, apply_dropout=True):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    if apply_dropout:
        D1 = np.random.rand(*A1.shape) < (1 - dropout_rate)
        A1 *= D1
    else:
        D1 = None

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    cache = {"A1": A1, "D1": D1, "A2": A2}
    return A2, cache

# Compute cross-entropy loss
def cross_entropy_loss(y, t):
    n = t.shape[0]
    log_likelihood = -np.log(y[range(n), t.argmax(axis=1)])
    return np.sum(log_likelihood) / n

# Backpropagation
def backpropagation(X, t, cache):
    A1, A2, D1 = cache["A1"], cache["A2"], cache["D1"]

    # Output layer
    dZ2 = A2 - t
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer error
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)

    if D1 is not None:
        dZ1 *= D1
        dZ1 /= (1 - dropout_rate)

    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients


# Update parameters
def update_parameters(W1, b1, W2, b2, grads, learning_rate):
    W1 -= learning_rate * grads["dW1"]
    b1 -= learning_rate * grads["db1"]
    W2 -= learning_rate * grads["dW2"]
    b2 -= learning_rate * grads["db2"]
    return W1, b1, W2, b2


# train function
def train(train_X, train_y_hot, epochs, batch_size, learning_rate):
    global W1, b1, W2, b2
    n = train_X.shape[0]

    for epoch in range(epochs):
        #shuffle
        indices = np.random.permutation(n)
        X, t = train_X[indices], train_y_hot[indices]

        loss = 0

        #mini batch training
        for start_idx in range(0, n, batch_size):
            end_idx = start_idx + batch_size
            X_batch, y_batch = X[start_idx:end_idx], t[start_idx:end_idx]

            y, cache = forward_propagation(X_batch, apply_dropout=True)
            loss = cross_entropy_loss(y, y_batch)
            grads = backpropagation(X_batch, y_batch, cache)

            W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, grads, learning_rate)

        val_predictions, _ = forward_propagation(train_X, apply_dropout=False)
        val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == np.argmax(train_y_hot, axis=1))

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")


# training
train(train_X, train_y_hot, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

# accuracy
final_predictions, _ = forward_propagation(test_X, apply_dropout=False)
final_accuracy = np.mean(np.argmax(final_predictions, axis=1) == np.argmax(test_y_hot, axis=1))
print(f"Final Validation Accuracy: {final_accuracy * 100:.2f}%")