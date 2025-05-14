# main.py
# 使用 One-vs-Rest QBoost 实现 0-3 手写数字多分类，并可视化结果与准确率、选中弱分类器数目随 K 变化

import numpy as np
import matplotlib.pyplot as plt
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
    H = np.vstack([clf.predict(X_train) for clf in stumps]).T.astype(float)
    for c in classes:
        print(f"Training QBoost for class {c} vs rest with M={M}")
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


def visualize_samples(X_test, y_test, y_pred, num_images=5, seed=42):
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(X_test), size=num_images, replace=False)
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(idxs):
        img = X_test[idx].reshape(8, 8)
        true = y_test[idx]
        pred = y_pred[idx]
        plt.subplot(1, num_images, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true}\nPred: {pred}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_accuracy_and_counts_vs_K(X_train, y_train, X_test, y_test, stumps, classes,
                                  K_values, lambda_reg=0.05, num_reads=200):
    """
    绘制准确率与总选中弱分类器数目随 K 变化的曲线图。
    """
    accuracies = []
    total_counts = []
    for K in K_values:
        classifiers = train_ovr_qboost(X_train, y_train, stumps, classes,
                                       M=K, lambda_reg=lambda_reg, num_reads=num_reads)
        y_pred = predict_ovr(classifiers, X_test)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)
        # 统计所选弱分类器总数
        count = sum(len(v) for v in classifiers.values())
        total_counts.append(count)
        print(f"K={K}, accuracy={acc*100:.2f}%, selected classifiers={count}")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(K_values, np.array(accuracies) * 100, 'o-', label='Accuracy (%)')
    ax2.plot(K_values, total_counts, 's--', label='Total Selected', color='gray')

    ax1.set_xlabel('K (Top weak classifiers)')
    ax1.set_ylabel('Accuracy (%)')
    ax2.set_ylabel('Total Selected Stumps')
    ax1.set_title('0-3 Classification: Accuracy and Selected Stumps vs K')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    ax1.grid(True)
    fig.tight_layout()
    plt.show()


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

    # 4. 一次性训练与评估
    classes = [0, 1, 2, 3]
    classifiers = train_ovr_qboost(X_train, y_train, stumps, classes,
                                   M=50, lambda_reg=0.05, num_reads=200)
    y_pred = predict_ovr(classifiers, X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"0-3 Multi-class QBoost OVR accuracy: {accuracy * 100:.2f}%")

    # 5. 可视化部分测试样本
    # visualize_samples(X_test, y_test, y_pred, num_images=5, seed=42)

    # 6. 绘制准确率与选中弱分类器数目随 K 变化图
    K_values = [10, 20, 30, 40, 50, 75, 100]
    plot_accuracy_and_counts_vs_K(X_train, y_train, X_test, y_test,
                                  stumps, classes, K_values,
                                  lambda_reg=0.05, num_reads=200)


if __name__ == '__main__':
    main()
