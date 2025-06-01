# main.py
# 使用 One-vs-Rest QBoost 实现 0-3 手写数字多分类（完整、修正版）

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from utility import generate_weak_classifiers, evaluate_stumps
from Ising import qboost_train


def train_ovr_qboost(X_train, y_train, stumps, classes,
                     M=200, lambda_reg=0.001, num_reads=500,
                     use_quantum=False, random_state=42):
    """
    One-vs-Rest QBoost 训练所有二分类子模型。
    返回: {class_label: (selected_stumps, threshold)}

    参数:
      - M: 初选弱分类器数目
      - lambda_reg: QUBO 正则化参数
      - num_reads: 退火读取次数
      - use_quantum: False 用经典 SA, True 用 SQA
    """
    rng = np.random.RandomState(random_state)
    classifiers = {}
    # 预计算 H 矩阵
    H_full = np.vstack([clf.predict(X_train) for clf in stumps]).T.astype(float)

    for c in classes:
        print(f"\nTraining QBoost for class {c} vs rest")
        # 二分类标签
        y_binary = np.where(y_train == c, 1, -1)
        # 等量采样
        pos_idx = np.where(y_binary == 1)[0]
        neg_idx = np.where(y_binary == -1)[0]
        N_pos = len(pos_idx)
        neg_sample = rng.choice(neg_idx, size=N_pos, replace=False)
        sel_idx = np.concatenate([pos_idx, neg_sample])
        X_bal = X_train[sel_idx]
        y_bal = y_binary[sel_idx]
        H = H_full[sel_idx]

        # 初选弱分类器
        scores0 = np.abs((H * y_bal[:, None]).sum(axis=0))
        top_idx = np.argsort(-scores0)[:M]
        top_stumps = [stumps[i] for i in top_idx]

        # 打印弱分类器分位数
        accs = evaluate_stumps(X_bal, y_bal, top_stumps)
        print("  弱分类器准确率分位数:", np.percentile(accs, [0,10,25,50,75,90,100]))

        # QBoost 选桩
        selected = qboost_train(
            top_stumps,
            X_bal,
            y_bal,
            lambda_reg=lambda_reg,
            num_reads=num_reads,
            use_quantum=use_quantum
        )

        # 计算并校准阈值
        if selected:
            raw = np.vstack([st.predict(X_bal) for st in selected]).T
            avg_score = raw.mean(axis=1)
            uniq = np.unique(avg_score)
            best_thr, best_acc = None, 0.0
            for thr in uniq:
                y_hat = np.where(avg_score >= thr, 1, -1)
                acc = (y_hat == y_bal).mean()
                if acc > best_acc:
                    best_acc, best_thr = acc, thr
            print(f"  校准阈值 thr={best_thr:.3f}, 训练准确率={best_acc*100:.2f}%")
            classifiers[c] = (selected, best_thr)
        else:
            print("  ⚠️  未选中任何弱分类器")
            classifiers[c] = ([], 0.0)

    return classifiers


def predict_ovr(classifiers, X):
    classes = list(classifiers.keys())
    margins = np.zeros((X.shape[0], len(classes)))
    for idx, c in enumerate(classes):
        selected, thr = classifiers[c]
        if not selected:
            margins[:, idx] = -np.inf
        else:
            raw = np.vstack([st.predict(X) for st in selected]).T
            avg_score = raw.mean(axis=1)
            margins[:, idx] = avg_score - thr
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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. 生成全局弱分类器
    stumps = generate_weak_classifiers(X_train, y_train)
    print(f"Generated {len(stumps)} weak classifiers for 0-3 classes")

    # 4. QBoost 训练
    classes = [0, 1, 2, 3]
    classifiers = train_ovr_qboost(
        X_train, y_train, stumps, classes,
        M=200, lambda_reg=0.001, num_reads=500,
        use_quantum=False
    )

    # 5. 多分类预测与评估
    y_pred = predict_ovr(classifiers, X_test)
    acc = np.mean(y_pred == y_test)
    print(f"0-3 Multi-class QBoost OVR accuracy: {acc * 100:.2f}%")


if __name__ == '__main__':
    main()