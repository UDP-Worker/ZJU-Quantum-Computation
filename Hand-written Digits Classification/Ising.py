
import numpy as np
import openjij as oj
from utility import load_data, generate_weak_classifiers, DecisionStump


def build_qubo_matrix(stumps, X, y, lambda_reg=0.01):

    N = len(y)
    M = len(stumps)
    # H[i,j] = h_j(x_i)
    H = np.vstack([clf.predict(X) for clf in stumps]).T
    G = H.T.dot(H)
    c = (H * y[:, None]).sum(axis=0)

    Q = {}
    C = 1.0 / (N**2)
    alpha = 2.0 / N
    for j in range(M):

        Q[(j, j)] = C * G[j, j] - alpha * c[j] + lambda_reg

        for k in range(j+1, M):
            qjk = C * G[j, k]
            if qjk != 0.0:
                Q[(j, k)] = qjk
    return Q


def qboost_train(stumps, X_train, y_train, lambda_reg=0.01, num_reads=100):
    Q = build_qubo_matrix(stumps, X_train, y_train, lambda_reg)
    sampler = oj.SQASampler()


    schedule = [
        [0.0, 0.1, 500],
        [1.0, 5.0, 500]
    ]


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

    agg = sum(clf.predict(X) for clf in stumps)
    return np.sign(agg)


def evaluate(stumps, X_test, y_test):
    pred = predict_strong(stumps, X_test)
    return (pred == y_test).mean()


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_data(test_size=0.2)

    all_stumps = generate_weak_classifiers(X_train, num_thresholds=10)

    scores = np.abs([(clf.predict(X_train) * y_train).sum() for clf in all_stumps])
    M = 200
    top_idx = np.argsort(-np.array(scores))[:M]
    stumps = [all_stumps[i] for i in top_idx]
    print(f"using {len(stumps)} weak classifiers for QBoost")


    selected = qboost_train(stumps, X_train, y_train, lambda_reg=0.01, num_reads=100)
    print(f" {len(selected)} weak classifiers selected")


    acc = evaluate(selected, X_test, y_test)
    print(f"Accuracy: {acc * 100:.2f}%")
