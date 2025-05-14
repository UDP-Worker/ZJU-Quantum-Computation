# benchmark.py
# 基准测试：随机选择若干弱分类器进行二分类，并输出结果与可视化

import numpy as np
import matplotlib.pyplot as plt
import random
from utility import load_data, generate_weak_classifiers


def main():
    # 1. 加载数据（二分类数字对）
    classes = (6, 9)  # 可修改为其它任意二分类对
    X_train, X_test, y_train, y_test = load_data(classes=classes, test_size=0.2)

    # 2. 生成所有弱分类器
    all_stumps = generate_weak_classifiers(X_train, num_thresholds=10)
    print(f"共生成 {len(all_stumps)} 个弱分类器")

    # 3. 随机选择 K 个弱分类器
    K = 3
    random.seed()
    selected = random.sample(all_stumps, K)
    print(f"随机选取 {K} 个弱分类器进行组合")
    for i, stump in enumerate(selected):
        print(f"Stump {i}: feature={stump.feature_index}, thr={stump.threshold:.3f}, pol={stump.polarity}")

    # 4. 基于随机弱分类器组合进行预测：累加预测值并取符号
    def predict_strong_random(stumps, X):
        agg = np.sum([clf.predict(X) for clf in stumps], axis=0)
        return np.sign(agg)

    preds = predict_strong_random(selected, X_test)
    acc = (preds == y_test).mean()
    print(f"测试集准确率: {acc * 100:.2f}%")

    # 5. 可视化前 10 个测试样本及其真值/预测标签
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
    fig.suptitle(f"随机弱分类器组合 ({classes[0]} vs {classes[1]}): 准确率 {acc*100:.2f}%")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()