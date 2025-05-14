# main.py
# 使用 One-vs-Rest QBoost 实现 0-3 手写数字多分类

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from utility import generate_weak_classifiers
from Ising import qboost_train


def train_ovr_qboost(X_train, y_train, stumps, classes, M=50, lambda_reg=0.05, num_reads=200):
    """
    One-vs-Rest QBoost 训练所有二分类子模型，返回：
      class_label -> 对应的强分类器列表
    """
    classifiers = {}
    # 预计算 H 矩阵
    H = np.vstack([clf.predict(X_train) for clf in stumps]).T.astype(float)

    for c in classes:
        print(f"Training QBoost for class {c} vs rest")
        y_binary = np.where(y_train == c, 1, -1)
        scores = np.abs((H * y_binary[:, None]).sum(axis=0))
        top_idx = np.argsort(-scores)[:M]
        top_stumps = [stumps[i] for i in top_idx]
        selected = qboost_train(top_stumps, X_train, y_binary,
                                lambda_reg=lambda_reg, num_reads=num_reads)
        classifiers[c] = selected
    return classifiers


def predict_ovr(classifiers, X):
    """
    多分类预测：对每个类累积分数，选最大者。
    """
    classes = list(classifiers.keys())
    margins = np.zeros((X.shape[0], len(classes)))
    for idx, c in enumerate(classes):
        for clf in classifiers[c]:
            margins[:, idx] += clf.predict(X)
    pred_idx = np.argmax(margins, axis=1)
    return np.array([classes[i] for i in pred_idx])


def main():
    # 1. 加载并筛选 0-3 类数据
    digits = load_digits()
    X, y = digits.data, digits.target
    mask = (y >= 0) & (y <= 3)
    X, y = X[mask], y[mask]

    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. 生成全局弱分类器
    stumps = generate_weak_classifiers(X_train, num_thresholds=10)
    print(f"Generated {len(stumps)} weak classifiers for 0-3 classes")

    # 4. One-vs-Rest QBoost 训练
    classes = [0, 1, 2, 3]
    classifiers = train_ovr_qboost(
        X_train, y_train, stumps, classes,
        M=50, lambda_reg=0.05, num_reads=200
    )

    # 5. 多分类预测与评估
    y_pred = predict_ovr(classifiers, X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"0-3 Multi-class QBoost OVR accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
