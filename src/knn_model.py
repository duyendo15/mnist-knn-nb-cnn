import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from data_loader import load_mnist, preprocess_for_classical

(X_train, y_train), (X_test, y_test) = load_mnist()
X_train_flat, X_test_flat = preprocess_for_classical(X_train, X_test)


# Dùng tập con để tuning k (tránh quá chậm)
np.random.seed(42)
subset_size = 5000
subset_idx = np.random.choice(len(X_train_flat), subset_size, replace=False)
X_train_subset = X_train_flat[subset_idx]
y_train_subset = y_train[subset_idx]

# Định nghĩa lưới tham số cần thử
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()

grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("Bắt đầu GridSearchCV cho KNN...")
start_time = time.time()
grid_search.fit(X_train_subset, y_train_subset)
tuning_time = time.time() - start_time

print(f"\nThời gian tuning: {tuning_time:.2f} giây")
print("Tham số tốt nhất:", grid_search.best_params_)
print("Accuracy tốt nhất (CV):", grid_search.best_score_)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Lấy model tốt nhất từ GridSearchCV
best_knn = grid_search.best_estimator_

# Train lại trên toàn bộ tập train (60,000 mẫu) -- không chỉ subset nữa
print("\nTrain lại KNN trên toàn bộ tập train...")
start_time = time.time()
best_knn.fit(X_train_flat, y_train)
train_time = time.time() - start_time
print(f"Thời gian train: {train_time:.2f} giây")

# Dự đoán trên tập test
print("Đang dự đoán trên tập test...")
start_time = time.time()
y_pred = best_knn.predict(X_test_flat)
predict_time = time.time() - start_time
print(f"Thời gian predict: {predict_time:.2f} giây")

# Đánh giá
test_accuracy = accuracy_score(y_test, y_pred)
test_f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\n=== KẾT QUẢ KNN TRÊN TẬP TEST ===")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1-macro: {test_f1_macro:.4f}")
print("\nBáo cáo chi tiết theo từng lớp:")
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
import seaborn as sns
import json

# Vẽ confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix - KNN")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.savefig("results/figures/knn_confusion_matrix.png")
plt.show()

# Lưu kết quả vào file JSON để tổng hợp so sánh sau này
knn_results = {
    "model": "KNN",
    "best_params": grid_search.best_params_,
    "cv_accuracy": grid_search.best_score_,
    "test_accuracy": test_accuracy,
    "test_f1_macro": test_f1_macro,
    "tuning_time_sec": tuning_time,
    "train_time_sec": train_time,
    "predict_time_sec": predict_time
}

with open("results/metrics/knn_results.json", "w") as f:
    json.dump(knn_results, f, indent=4)

print("\nĐã lưu kết quả KNN vào results/metrics/knn_results.json")