import os
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from data_loader import load_mnist, preprocess_for_cnn

(X_train, y_train), (X_test, y_test) = load_mnist()
X_train_cnn, X_test_cnn = preprocess_for_cnn(X_train, X_test)

print("X_train_cnn shape:", X_train_cnn.shape)
print("y_train shape:", y_train.shape)

def build_cnn_model():
    """
    Xây dựng kiến trúc CNN cho bài toán phân loại MNIST (10 lớp).
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


model_path = "models/cnn_model.keras"

if os.path.exists(model_path):
    print(f"\nĐã tìm thấy model đã train tại {model_path}, load lại...")
    cnn = tf.keras.models.load_model(model_path)
    cnn.summary()
    train_time = None
    history = None
else:
    print("\nChưa có model đã lưu, xây dựng và train CNN...")
    cnn = build_cnn_model()
    cnn.summary()

    print("\nBắt đầu train CNN...")
    start_time = time.time()

    history = cnn.fit(
        X_train_cnn, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    train_time = time.time() - start_time
    print(f"\nThời gian train: {train_time:.2f} giây")

    # Lưu model lại để lần sau không cần train nữa
    cnn.save(model_path)
    print(f"Đã lưu model vào {model_path}")


# Chỉ vẽ learning curves khi vừa train xong (không có khi load model)
if history is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Accuracy theo Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Loss theo Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/figures/cnn_learning_curves.png")
    plt.show()

    print("\nĐang dự đoán trên tập test...")
start_time = time.time()
y_pred_proba = cnn.predict(X_test_cnn)
predict_time = time.time() - start_time
y_pred = np.argmax(y_pred_proba, axis=1)

print(f"Thời gian predict: {predict_time:.2f} giây")

test_accuracy = accuracy_score(y_test, y_pred)
test_f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\n=== KẾT QUẢ CNN TRÊN TẬP TEST ===")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1-macro: {test_f1_macro:.4f}")
print("\nBáo cáo chi tiết theo từng lớp:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix - CNN")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.savefig("results/figures/cnn_confusion_matrix.png")
plt.show()

# Lưu kết quả (xử lý trường hợp train_time/history có thể là None nếu load model)
cnn_results = {
    "model": "CNN",
    "total_params": int(cnn.count_params()),
    "epochs": 10,
    "test_accuracy": test_accuracy,
    "test_f1_macro": test_f1_macro,
    "train_time_sec": train_time,
    "predict_time_sec": predict_time
}

if history is not None:
    cnn_results["final_train_accuracy"] = float(history.history['accuracy'][-1])
    cnn_results["final_val_accuracy"] = float(history.history['val_accuracy'][-1])

with open("results/metrics/cnn_results.json", "w") as f:
    json.dump(cnn_results, f, indent=4)

print("\nĐã lưu kết quả CNN vào results/metrics/cnn_results.json")