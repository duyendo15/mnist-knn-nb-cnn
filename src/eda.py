import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from data_loader import load_mnist

def plot_class_distribution(y_train, y_test):
    """
    Vẽ biểu đồ phân bố số lượng mẫu theo từng lớp (0-9)
    trong tập train và test, để kiểm tra dữ liệu có cân bằng không.
    """
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    digits = sorted(train_counts.keys())
    train_values = [train_counts[d] for d in digits]
    test_values = [test_counts[d] for d in digits]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(digits, train_values, color='steelblue')
    axes[0].set_title("Phân bố lớp - Tập Train")
    axes[0].set_xlabel("Chữ số")
    axes[0].set_ylabel("Số lượng mẫu")
    axes[0].set_xticks(digits)

    axes[1].bar(digits, test_values, color='coral')
    axes[1].set_title("Phân bố lớp - Tập Test")
    axes[1].set_xlabel("Chữ số")
    axes[1].set_xticks(digits)

    plt.tight_layout()
    plt.savefig("results/figures/class_distribution.png")
    plt.show()

    print("Số lượng mẫu mỗi lớp (Train):", dict(sorted(train_counts.items())))
    print("Số lượng mẫu mỗi lớp (Test):", dict(sorted(test_counts.items())))

def plot_sample_images(X_train, y_train, n_per_class=1):
    """
    Hiển thị ảnh mẫu cho mỗi chữ số (0-9) để xem trực quan dữ liệu.
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for digit in range(10):
        idx = np.where(y_train == digit)[0][0]  # lấy ảnh đầu tiên của chữ số này
        ax = axes[digit // 5, digit % 5]
        ax.imshow(X_train[idx], cmap='gray')
        ax.set_title(f"Nhãn: {digit}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("results/figures/sample_images.png")
    plt.show()
if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()
    plot_class_distribution(y_train, y_test)
    plot_sample_images(X_train, y_train)