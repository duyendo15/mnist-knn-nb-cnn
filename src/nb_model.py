import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from data_loader import load_mnist, preprocess_for_classical

(X_train, y_train), (X_test, y_test) = load_mnist()
X_train_flat, X_test_flat = preprocess_for_classical(X_train, X_test)

# Tuning: GaussianNB chỉ có 1 tham số chính đáng để tune là var_smoothing
param_grid = {
    'var_smoothing': np.logspace(0, -9, num=10)
}

nb = GaussianNB()
grid_search = GridSearchCV(
    estimator=nb,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("Bắt đầu GridSearchCV cho Naive Bayes...")
start_time = time.time()
grid_search.fit(X_train_flat, y_train)   # NB nhanh, dùng luôn toàn bộ train, không cần subset
tuning_time = time.time() - start_time

print(f"\nThời gian tuning: {tuning_time:.2f} giây")
print("Tham số tốt nhất:", grid_search.best_params_)
print("Accuracy tốt nhất (CV):", grid_search.best_score_)

best_nb = grid_search.best_estimator_

print("\nTrain lại Naive Bayes trên toàn bộ tập train...")
start_time = time.time()
best_nb.fit(X_train_flat, y_train)
train_time = time.time() - start_time
print(f"Thời gian train: {train_time:.2f} giây")

print("Đang dự đoán trên tập test...")
start_time = time.time()
y_pred = best_nb.predict(X_test_flat)
predict_time = time.time() - start_time
print(f"Thời gian predict: {predict_time:.2f} giây")

test_accuracy = accuracy_score(y_test, y_pred)
test_f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"\n=== KẾT QUẢ NAIVE BAYES TRÊN TẬP TEST ===")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1-macro: {test_f1_macro:.4f}")
print("\nBáo cáo chi tiết theo từng lớp:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.tight_layout()
plt.savefig("results/figures/nb_confusion_matrix.png")
plt.show()

# Lưu kết quả
nb_results = {
    "model": "Naive Bayes",
    "best_params": {"var_smoothing": float(grid_search.best_params_['var_smoothing'])},
    "cv_accuracy": grid_search.best_score_,
    "test_accuracy": test_accuracy,
    "test_f1_macro": test_f1_macro,
    "tuning_time_sec": tuning_time,
    "train_time_sec": train_time,
    "predict_time_sec": predict_time
}

with open("results/metrics/nb_results.json", "w") as f:
    json.dump(nb_results, f, indent=4)

print("\nĐã lưu kết quả Naive Bayes vào results/metrics/nb_results.json")