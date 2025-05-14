import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. 加载 digits 数据集
digits = load_digits()
X, y = digits.data, digits.target

# 2. 选择一个特征索引，这里以像素 (4,4) 为例
feature_row, feature_col = 4, 4
feature_idx = feature_row * 8 + feature_col
values = X[:, feature_idx]

# 3. 将样本按该特征值排序
sorted_idx = np.argsort(values)
N = len(values)
group_size = N // 5

# 4. 每个区间取前 10 个样本索引
sample_indices = []
for i in range(5):
    start = i * group_size
    end = (i + 1) * group_size if i < 4 else N
    group_idx = sorted_idx[start:end]
    sample_indices.append(group_idx[:10])

labels = ["Very small", "Small", "Medium", "Large", "Very large"]

# 5. 绘制 5 行 10 列的图像网格
fig, axes = plt.subplots(5, 10, figsize=(15, 8))
for i in range(5):
    for j in range(10):
        idx = sample_indices[i][j]
        img = X[idx].reshape(8, 8)
        ax = axes[i, j]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if j == 0:
            # 标注每行分组
            ax.set_ylabel(labels[i], rotation=0, fontsize=12, labelpad=40, va='center')
    # 在行首上方加入对应特征值（第一个样本）
    axes[i, 0].set_title(f"val={values[sample_indices[i][0]]:.1f}", fontsize=10)

fig.suptitle(f"Feature pixel_{feature_row}_{feature_col} (index={feature_idx}) — 每区 10 张样本", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
