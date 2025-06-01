# demo.py
# 演示 QBoost 强分类器对手写数字二分类（如 0 vs 1）的效果

import numpy as np
import matplotlib.pyplot as plt
from utility import load_data, generate_weak_classifiers
from Ising import qboost_train, predict_strong


def main():

    classes = (4, 6)
    X_train, X_test, y_train, y_test = load_data(classes=classes, test_size=0.2)


    all_stumps = generate_weak_classifiers(X_train, num_thresholds=10)

    scores = np.abs([(clf.predict(X_train) * y_train).sum() for clf in all_stumps])
    M = 50
    top_idx = np.argsort(-np.array(scores))[:M]
    stumps = [all_stumps[i] for i in top_idx]
    print(f"use {len(stumps)} weak classifiers for QBoost")


    selected = qboost_train(stumps, X_train, y_train, lambda_reg=0.01, num_reads=50)
    print(f"{len(selected)} weak classifiers selected")
    for i, stump in enumerate(selected):
        print(f"Stump {i}: feature={stump.feature_index}, thr={stump.threshold:.3f}, pol={stump.polarity}")


    preds = predict_strong(selected, X_test)
    acc = (preds == y_test).mean()
    print(f"Test set accuracy: {acc * 100:.2f}%")

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
    fig.suptitle(f"QBoost classification ({classes[0]} vs {classes[1]}): Accuracy: {acc*100:.2f}%")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
