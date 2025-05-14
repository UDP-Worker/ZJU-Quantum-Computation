# Ising.py
# QBoost 实现：利用 OpenJij 模拟量子退火（SQA）构建强分类器

import numpy as np
import openjij as oj
from utility import load_data, generate_weak_classifiers, DecisionStump


def build_qubo_matrix(stumps, X, y, lambda_reg=0.01):
    """
    构建 QBoost 的 QUBO 矩阵，将弱分类器的组合问题映射为二次无约束二元优化问题。

    Q[j,j] = C * G[j,j] - alpha * c[j] + lambda_reg
    Q[j,k] = C * G[j,k]  (j<k)

    其中:
      - G[j,k] = sum_i h_j(x_i) h_k(x_i)
      - c[j]   = sum_i y_i h_j(x_i)
      - C = 1 / N^2, alpha = 2 / N
    """
    N = len(y)
    M = len(stumps)
    # H[i,j] = h_j(x_i)
    H = np.vstack([clf.predict(X) for clf in stumps]).T  # shape (N, M)
    G = H.T.dot(H)                                      # shape (M, M)
    c = (H * y[:, None]).sum(axis=0)                    # shape (M,)

    Q = {}
    C = 1.0 / (N**2)
    alpha = 2.0 / N
    for j in range(M):
        # 对角项
        Q[(j, j)] = C * G[j, j] - alpha * c[j] + lambda_reg
        # 非对角项
        for k in range(j+1, M):
            qjk = C * G[j, k]
            if qjk != 0.0:
                Q[(j, k)] = qjk
    return Q


def qboost_train(stumps, X_train, y_train, lambda_reg=0.01, num_reads=100):
    Q = build_qubo_matrix(stumps, X_train, y_train, lambda_reg)
    sampler = oj.SQASampler()

    # 修正后的 schedule：每行 [s, beta, sweeps]
    schedule = [
        [0.0, 0.1, 500],  # s=0 时 β=0.1，做 500 步
        [1.0, 5.0, 500]   # s=1 时 β=5.0，做 500 步
    ]

    # 同时指定 trotter 切片数（可选，默认为 1）
    response = sampler.sample_qubo(
        Q,
        num_reads=num_reads,
        trotter=8,
        schedule=schedule
    )

    best = response.first.sample
    sel_idx = [j for j, bit in best.items() if bit == 1]
    return [stumps[j] for j in sel_idx]


def predict_strong(stumps, X):
    """
    对测试数据 X 应用强分类器，返回 +1/-1 预测。
    """
    # 所有弱分类器预测值累加，取符号
    agg = sum(clf.predict(X) for clf in stumps)
    return np.sign(agg)


def evaluate(stumps, X_test, y_test):
    pred = predict_strong(stumps, X_test)
    return (pred == y_test).mean()


if __name__ == '__main__':
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_data(test_size=0.2)
    # 2. 生成弱分类器并筛选 top-K
    all_stumps = generate_weak_classifiers(X_train, num_thresholds=10)
    # 根据与标签的相关性排序并选前 M 个([quantumcomputinginc.com](https://quantumcomputinginc.com/learn/module/understanding-qubos/qboost-formulation?utm_source=chatgpt.com))
    scores = np.abs([(clf.predict(X_train) * y_train).sum() for clf in all_stumps])
    M = 200
    top_idx = np.argsort(-np.array(scores))[:M]
    stumps = [all_stumps[i] for i in top_idx]
    print(f"使用 {len(stumps)} 个弱分类器进行 QBoost")

    # 3. 训练 QBoost
    selected = qboost_train(stumps, X_train, y_train, lambda_reg=0.01, num_reads=100)
    print(f"选中了 {len(selected)} 个弱分类器构建强分类器")

    # 4. 测试评估
    acc = evaluate(selected, X_test, y_test)
    print(f"测试集准确率: {acc * 100:.2f}%")
