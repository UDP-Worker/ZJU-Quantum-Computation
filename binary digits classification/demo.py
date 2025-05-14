# demo.py
# 演示 QBoost 强分类器对手写数字二分类（如 0 vs 1）的效果

import numpy as np
import matplotlib.pyplot as plt
from utility import load_data, generate_weak_classifiers
from Ising import qboost_train, predict_strong


def main():
    # 1. 加载数据（可自定义分类对）
    classes = (3, 8)  # 二分类数字，例如 (3, 8)
    X_train, X_test, y_train, y_test = load_data(classes=classes, test_size=0.2)

    # 2. 生成弱分类器并筛选前 M 个
    all_stumps = generate_weak_classifiers(X_train, num_thresholds=10)
    # 按与标签相关性排序
    scores = np.abs([(clf.predict(X_train) * y_train).sum() for clf in all_stumps])
    M = 50
    top_idx = np.argsort(-np.array(scores))[:M]
    stumps = [all_stumps[i] for i in top_idx]
    print(f"使用 {len(stumps)} 个弱分类器进行 QBoost")

    # 3. 训练 QBoost，选出强分类器中的弱分类器子集
    selected = qboost_train(stumps, X_train, y_train, lambda_reg=0.01, num_reads=50)
    print(f"选中了 {len(selected)} 个弱分类器构建强分类器")
    for i, stump in enumerate(selected):
        print(f"Stump {i}: feature={stump.feature_index}, thr={stump.threshold:.3f}, pol={stump.polarity}")

    # 4. 在测试集上进行预测
    preds = predict_strong(selected, X_test)
    acc = (preds == y_test).mean()
    print(f"测试集准确率: {acc * 100:.2f}%")

    # 5. 可视化：展示前 10 个测试样本的图像与分类结果
    num_show = 10
    fig, axes = plt.subplots(2, num_show//2, figsize=(12, 5))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        img = X_test[idx].reshape(8, 8)
        true_label = y_test[idx]
        pred_label = preds[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"T:{int((true_label+1)/2)} P:{int((pred_label+1)/2)}")
        ax.axis('off')
    fig.suptitle(f"QBoost 二分类 ({classes[0]} vs {classes[1]}): 准确率 {acc*100:.2f}%")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
