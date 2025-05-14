# Ising.py
# 改进版 QBoost：修正 QUBO 缩放、退火调度与正则化
# 2025-05-14

import numpy as np
import openjij as oj
from utility import DecisionStump

# ------------------------------------------------------------------------------------
# QBoost 关键函数
# ------------------------------------------------------------------------------------

def build_qubo_matrix(stumps, X, y, lambda_reg=0.05):
    """构建 QBoost QUBO (改进版)

    原始 QBoost 论文 (Khoshaman et al., 2018) 使用的系数为
        C = 1/N,  α = 2/N  (N = 样本数)
    旧实现将 C 设成 1/N²，导致信息缩放过小、正则化占主导 → 训练失败。
    此处恢复为 C = 1/N。
    """
    N = len(y)
    M = len(stumps)
    # H[i,j] = h_j(x_i)
    H = np.vstack([clf.predict(X) for clf in stumps]).T.astype(float)  # (N, M)
    # Gram & correlation 向量
    G = H.T @ H                       # (M, M)
    c = (H * y[:, None]).sum(axis=0)  # (M,)

    # QUBO 构造
    Q = {}
    C = 1.0 / N       # 核心缩放系数
    alpha = 2.0 / N

    for j in range(M):
        Q[(j, j)] = C * G[j, j] - alpha * c[j] + lambda_reg
        for k in range(j + 1, M):
            qjk = C * G[j, k]
            if qjk != 0.0:
                Q[(j, k)] = qjk
    return Q


def qboost_train(stumps, X_train, y_train,
                 lambda_reg=0.05,
                 num_reads=200,
                 trotter=4):
    """使用 SQA 对 QUBO 求解并返回被选中的弱分类器列表 (bit==1)。"""
    Q = build_qubo_matrix(stumps, X_train, y_train, lambda_reg)

    # 退火调度：s ∈ [0,1]
    schedule = [
        [0.0, 0.1, 800],   # 起始：低 β，高温
        [1.0, 6.0, 800]    # 结束：高 β，接近基态
    ]

    sampler = oj.SQASampler()
    response = sampler.sample_qubo(
        Q,
        num_reads=num_reads,
        trotter=trotter,
        schedule=schedule
    )

    best = response.first.sample
    sel_idx = [j for j, bit in best.items() if bit == 1]
    return [stumps[j] for j in sel_idx]


# ------------------------------------------------------------------------------------
# 推断辅助
# ------------------------------------------------------------------------------------

def predict_strong(stumps, X):
    """聚合弱分类器预测 → sign(sum)。若无弱分类器则返回全 -1。"""
    if not stumps:
        return -np.ones(X.shape[0])
    agg = sum(clf.predict(X) for clf in stumps)
    return np.sign(agg)


def evaluate(stumps, X_test, y_test):
    pred = predict_strong(stumps, X_test)
    return (pred == y_test).mean()


# ------------------------------------------------------------------------------------
# 单元测试入口（可选）
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    from utility import load_data, generate_weak_classifiers

    X_train, X_test, y_train, y_test = load_data(classes=(0, 1))
    all_stumps = generate_weak_classifiers(X_train, num_thresholds=15)

    # 相关性排序取 Top-K
    scores = np.abs([(clf.predict(X_train) * y_train).sum() for clf in all_stumps])
    K = 200
    stumps = [all_stumps[i] for i in np.argsort(-scores)[:K]]

    selected = qboost_train(stumps, X_train, y_train)
    acc = evaluate(selected, X_test, y_test)
    print(f"测试集准确率: {acc*100:.2f}% | 选中弱分类器 {len(selected)}/{K}")