import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 文件夹路径
folder_path = "./result"

# 获取文件夹中所有的 CSV 文件路径
file_paths = glob.glob(os.path.join(folder_path, "*.csv"))

# 模型名称列表和最大验证准确率列表
model_names = []
max_val_accuracy = []

# 颜色列表
colors = plt.cm.tab10(np.linspace(0, 1, len(file_paths)))

# 遍历每个 CSV 文件
for idx, file_path in enumerate(file_paths):
    # 提取模型名称
    model_name = os.path.splitext(os.path.basename(file_path))[0].split("_")[0]
    model_names.append(model_name)
    
    # 加载 CSV 文件
    df = pd.read_csv(file_path)
    
    # 提取最大验证准确率
    max_val_acc = df["val_accuracy"].max()
    max_val_accuracy.append(max_val_acc)

    # 绘制条形图
    plt.bar(model_name, max_val_acc, color=colors[idx])
    plt.text(model_name, max_val_acc, f"{max_val_acc:.5f}", ha="center", va="bottom")

# 添加图例和标签
plt.xlabel("Model")
plt.ylabel("Max Validation Accuracy")
plt.title("Comparison of Max Validation Accuracy among Models")
plt.legend(model_names, loc="lower right",fontsize="small")
plt.xticks(rotation=45)  # 旋转横坐标标签，使其更清晰
plt.tight_layout()  # 调整布局，防止标签重叠
plt.savefig("max_val_accuracy_comparison.png")
plt.show()