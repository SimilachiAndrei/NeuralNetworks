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

train_X, train_Y = download_mnist(is_train=True)
test_X, test_Y = download_mnist(is_train=False)

#data normalization
train_X = train_X / 255.0
test_X = test_X / 255.0

#one-host encoding
def get_one_hot_encoding(y,size = 10):
    return np.eye(size)[y]

train_y_hot = get_one_hot_encoding(train_Y)
test_y_hot = get_one_hot_encoding(test_Y)

input_size = 784
output_size = 10

W = np.zeros((input_size, output_size))
b = np.zeros((output_size,))

def softmax(wsum):
    exp = np.exp(wsum)
    return exp / np.sum(exp, axis=1, keepdims=True)


def fw_propagation(X,W, b):
    z = np.dot(X, W) + b
    return softmax(z)

def predict(X, W, b):
    y = fw_propagation(X, W, b)
    return np.argmax(y, axis=1)

def cross_entropy(y_hat, y_true):
    entr = -np.sum(y_true * np.log(y_hat))
    return entr


def gradient_descent(X, y_hat, y_true, W, b, learning_rate):
    dz = y_true - y_hat

    dW = np.dot(X.T, dz)
    db = np.sum(dz, axis=0)

    W += learning_rate * dW
    b += learning_rate * db

    return W, b


def train_perceptron(train_X, train_Y_oh, W, b, epochs=100, batch_size=100, learning_rate=0.01):
    n_samples = train_X.shape[0]

    for epoch in range(epochs):
        permutation = np.random.permutation(n_samples)
        train_X = train_X[permutation]
        train_Y_oh = train_Y_oh[permutation]

        for i in range(0, n_samples, batch_size):
            X_batch = train_X[i:i + batch_size]
            y_batch = train_Y_oh[i:i + batch_size]

            y_hat = fw_propagation(X_batch, W, b)

            loss = cross_entropy(y_hat, y_batch)

            W, b = gradient_descent(X_batch, y_hat, y_batch, W, b, learning_rate)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')


train_perceptron(train_X, train_y_hot, W, b, epochs=50, batch_size=100, learning_rate=0.01)

test_predictions = predict(test_X, W, b)

accuracy = np.mean(test_predictions == test_Y)
print(f'Test Accuracy: {accuracy * 100:.2f}%')