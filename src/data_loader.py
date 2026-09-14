import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist():
    """
    Tải bộ dữ liệu MNIST (chữ số viết tay).
    Trả về: (X_train, y_train), (X_test, y_test)
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)

def preprocess_for_classical(X_train, X_test):
    """
    Tiền xử lý cho KNN và Naive Bayes:
    - Flatten ảnh 28x28 -> vector 784 chiều
    - Chuẩn hóa pixel về [0, 1]
    """
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    return X_train_flat, X_test_flat


def preprocess_for_cnn(X_train, X_test):
    """
    Tiền xử lý cho CNN:
    - Giữ dạng 2D, thêm chiều kênh (channel) = 1 vì ảnh xám
    - Chuẩn hóa pixel về [0, 1]
    """
    X_train_cnn = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test_cnn = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return X_train_cnn, X_test_cnn

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()

    X_train_flat, X_test_flat = preprocess_for_classical(X_train, X_test)
    print("Classical - X_train shape:", X_train_flat.shape)
    print("Classical - min/max:", X_train_flat.min(), "/", X_train_flat.max())

    X_train_cnn, X_test_cnn = preprocess_for_cnn(X_train, X_test)
    print("CNN - X_train shape:", X_train_cnn.shape)
    print("CNN - min/max:", X_train_cnn.min(), "/", X_train_cnn.max())