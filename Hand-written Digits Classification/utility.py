# utility.py
# 工具模块：用于加载 任意两类 手写数字数据集，并生成弱分类器（决策树桩）

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class DecisionStump:
    """
    决策桩：基于单一特征的阈值判断
    属性：
      - feature_index: 特征索引
      - threshold: 阈值
      - polarity: 极性（1 表示特征值 < 阈值时输出 -1，否则 +1；-1 则相反）
    """
    def __init__(self, feature_index: int, threshold: float, polarity: int = 1):
        self.feature_index = feature_index
        self.threshold = threshold
        self.polarity = polarity

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对输入数据 X（形状 [n_samples, n_features]）进行预测，返回 -1 或 +1
        """
        feature_values = X[:, self.feature_index]
        preds = np.ones(len(feature_values), dtype=int)
        if self.polarity == 1:
            preds[feature_values < self.threshold] = -1
        else:
            preds[feature_values > self.threshold] = -1
        return preds


def load_data(classes=(0, 1), test_size: float = 0.2, random_state: int = 42):
    """
    加载 sklearn 的 digits 数据集，筛选标签为 classes 中的两类样本，
    将第一个类映射为 -1，第二个类映射为 +1，
    并切分为训练集和测试集。

    参数:
      - classes: 二分类类别的元组，例如 (3, 8)
      - test_size: 测试集比例
      - random_state: 随机种子

    返回:
      X_train, X_test, y_train, y_test
    """
    digits = load_digits()
    X = digits.data
    y = digits.target
    class_a, class_b = classes
    mask = (y == class_a) | (y == class_b)
    X = X[mask]
    y = y[mask]
    # 标签映射：class_a -> -1, class_b -> +1
    y = np.where(y == class_a, -1, 1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_weak_classifiers(X: np.ndarray, num_thresholds: int = 10):
    """
    根据训练数据 X，针对每个特征按 num_thresholds 个等距阈值生成一系列决策桩。

    参数:
      - X: 训练特征数组，形状 [n_samples, n_features]
      - num_thresholds: 每个特征上生成的阈值数目

    返回:
      弱分类器列表，每个元素都是 DecisionStump 实例
    """
    stumps = []
    n_features = X.shape[1]
    for feature_idx in range(n_features):
        values = X[:, feature_idx]
        min_v, max_v = values.min(), values.max()
        thresholds = np.linspace(min_v, max_v, num_thresholds)
        for thresh in thresholds:
            stumps.append(DecisionStump(feature_idx, thresh, polarity=1))
            stumps.append(DecisionStump(feature_idx, thresh, polarity=-1))
    return stumps


if __name__ == "__main__":
    # 简单测试
    # 比如想二分类数字 3 和 8，则调用：
    X_train, X_test, y_train, y_test = load_data(classes=(6, 9))
    classifiers = generate_weak_classifiers(X_train, num_thresholds=10)
    print(f"生成了 {len(classifiers)} 个弱分类器（决策桩）")
