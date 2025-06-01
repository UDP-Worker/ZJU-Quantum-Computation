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


def qboost_train(
        stumps,
        X_train,
        y_train,
        lambda_reg=0.001,
        num_reads=500,
        use_quantum=False,          # ← True = SQA, False = SA
        trotter=8,
        random_state=42):
    """
    训练一个 QBoost 子模型, 返回被挑选出的弱分类器列表
    ----------------------------------------------------
    use_quantum=False : 经典模拟退火  (oj.SASampler)
    use_quantum=True  : 模拟量子退火 (oj.SQASampler)
    """

    # --- 1. 构造 QUBO ---
    Q = build_qubo_matrix(stumps, X_train, y_train, lambda_reg)

    # --- 2. 打印 Q 统计 ---
    if isinstance(Q, dict):        # OpenJij 允许 dict 或 ndarray
        M = max(max(i, j) for i, j in Q.keys()) + 1
        Q_arr = np.zeros((M, M))
        for (i, j), v in Q.items():
            Q_arr[i, j] = v
            if i != j:
                Q_arr[j, i] = v
    else:
        Q_arr = Q

    diag = np.diag(Q_arr)
    off  = Q_arr[np.triu_indices_from(Q_arr, k=1)]
    neg  = (diag < 0).sum()
    print(f"    ▶ Q diag min = {diag.min():.3g} | negative count = {neg}")
    print(f"    ▶ Q diag mean/std = {diag.mean():.3g} / {diag.std():.3g}")
    print(f"    ▶ Q off  mean/std = {off.mean():.3g} / {off.std():.3g}")

    # --- 3. 选择采样器 & schedule ---
    rng = np.random.RandomState(random_state)

    if use_quantum:                              # ===== SQA =====
        sampler = oj.SQASampler()
        # schedule 行格式: [ s , γ , sweeps ]
        sqa_schedule = [
            [0.0, 1.5, 4000],    # 强横场起步
            [1.0, 0.0, 4000]     # 衰减到 γ=0
        ]
        response = sampler.sample_qubo(
            Q,
            num_reads=num_reads,
            trotter=trotter,
            schedule=sqa_schedule,
            seed=int(rng.randint(2**31))
        )

    else:                                         # ===== SA =====
        sampler = oj.SASampler()
        # schedule 行格式: [ β , sweeps ]
        sa_schedule = [
            [0.1, 4000],         # 高温 (β 小) – 探索
            [5.0, 4000]          # 低温 (β 大) – 收敛
        ]
        response = sampler.sample_qubo(
            Q,
            num_reads=num_reads,
            schedule=sa_schedule,
            seed=int(rng.randint(2**31))
        )

    # --- 4. 解析最优样本 ---
    best = response.first.sample          # dict: {idx:0/1, ...}
    sel_idx = [j for j, bit in best.items() if bit == 1]

    # 可选：若完全没挑中桩，可打印提示
    if not sel_idx:
        print("    ⚠️  No stumps selected (all zeros)")

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
