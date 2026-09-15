import json
import matplotlib.pyplot as plt
import pandas as pd

# Đọc lại kết quả đã lưu của 3 thuật toán
with open("results/metrics/knn_results.json") as f:
    knn_results = json.load(f)

with open("results/metrics/nb_results.json") as f:
    nb_results = json.load(f)

with open("results/metrics/cnn_results.json") as f:
    cnn_results = json.load(f)

# Tạo bảng so sánh
comparison = pd.DataFrame([
    {
        "Model": "KNN",
        "Test Accuracy": knn_results["test_accuracy"],
        "F1-macro": knn_results["test_f1_macro"],
        "Train Time (s)": knn_results["train_time_sec"],
        "Predict Time (s)": knn_results["predict_time_sec"]
    },
    {
        "Model": "Naive Bayes",
        "Test Accuracy": nb_results["test_accuracy"],
        "F1-macro": nb_results["test_f1_macro"],
        "Train Time (s)": nb_results["train_time_sec"],
        "Predict Time (s)": nb_results["predict_time_sec"]
    },
    {
        "Model": "CNN",
        "Test Accuracy": cnn_results["test_accuracy"],
        "F1-macro": cnn_results["test_f1_macro"],
        "Train Time (s)": 150.79,  # ghi cứng vì lần chạy load model có train_time = None
        "Predict Time (s)": cnn_results["predict_time_sec"]
    }
])

print(comparison.to_string(index=False))
comparison.to_csv("results/metrics/comparison_table.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Biểu đồ 1: So sánh Accuracy & F1
x = comparison["Model"]
axes[0].bar(x, comparison["Test Accuracy"], color='steelblue', width=0.4, label='Accuracy', align='edge')
axes[0].bar(x, comparison["F1-macro"], color='coral', width=-0.4, label='F1-macro', align='edge')
axes[0].set_title("So sánh Accuracy & F1-macro")
axes[0].set_ylim(0, 1)
axes[0].legend()

# Biểu đồ 2: So sánh Train Time (log scale vì chênh lệch quá lớn)
axes[1].bar(x, comparison["Train Time (s)"], color='seagreen')
axes[1].set_title("So sánh Thời gian Train (giây)")
axes[1].set_yscale('log')

# Biểu đồ 3: So sánh Predict Time
axes[2].bar(x, comparison["Predict Time (s)"], color='darkorange')
axes[2].set_title("So sánh Thời gian Predict (giây)")

plt.tight_layout()
plt.savefig("results/figures/comparison_charts.png")
plt.show()