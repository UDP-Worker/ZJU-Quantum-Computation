import numpy as np
import matplotlib.pyplot as plt
from utility import load_data, generate_weak_classifiers
from Ising import qboost_train, predict_strong

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

def evaluate_random_guess(y_test, n_runs=1):
    """
    随机猜测基准：在 n_runs 次重复中，每次都随机选择标签，并计算平均准确率
    y_test: array of shape (n_samples,), 取值为 +1 或 -1
    """
    accuracies = []
    n_samples = len(y_test)
    for _ in range(n_runs):
        # 随机生成 ±1 预测
        preds = np.random.choice([-1, 1], size=n_samples)
        acc = (preds == y_test).mean()
        accuracies.append(acc)
    return np.mean(accuracies), np.std(accuracies)


def evaluate_classifiers(classes=(4, 6),
                         test_size=0.2,
                         n_repeats=5,
                         M=50,
                         lambda_reg=0.01,
                         num_reads=50,
                         random_guess_runs=1):
    """
    对比 QBoost、随机猜测基准和其他常见分类器在二分类手写数字任务上的平均准确率。

    参数：
    - classes: 二分类的数字对，例如 (4, 6)
    - test_size: 测试集比例
    - n_repeats: 重复的随机划分次数，用以统计平均准确率
    - M: 选择前 M 个弱分类器作为 QBoost 的候选
    - lambda_reg: QBoost 的正则化参数
    - num_reads: QBoost 的采样次数
    - random_guess_runs: 随机猜测基准自身的重复次数

    返回：
    一个字典，键为算法名称，值为 (平均准确率, 标准差)
    """
    # 用于存储每次重复的结果
    results = {
        "Random Guess": [],
        "Logistic Regression": [],
        "Decision Tree": [],
        "Random Forest": [],
        "SVM": [],
        "AdaBoost (SKLearn)": [],
        "QBoost": []
    }

    for repeat in range(n_repeats):
        # 每次使用不同的随机划分（由 load_data 内部控制随机种子）
        X_train, X_test, y_train, y_test = load_data(classes=classes, test_size=test_size)

        # SKLearn 需要将标签转换为 {0, 1} 格式
        y_train_skl = ((y_train + 1) // 2).astype(int)
        y_test_skl = ((y_test + 1) // 2).astype(int)

        # ---------- 随机猜测基准 ----------
        rg_mean, rg_std = evaluate_random_guess(y_test, n_runs=random_guess_runs)
        results["Random Guess"].append(rg_mean)

        # ---------- 经典机器学习分类器 ----------
        # 1. Logistic Regression
        lr = LogisticRegression(solver="liblinear")
        lr.fit(X_train, y_train_skl)
        preds_lr = lr.predict(X_test)
        acc_lr = accuracy_score(y_test_skl, preds_lr)
        results["Logistic Regression"].append(acc_lr)

        # 2. Decision Tree
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train_skl)
        preds_dt = dt.predict(X_test)
        acc_dt = accuracy_score(y_test_skl, preds_dt)
        results["Decision Tree"].append(acc_dt)

        # 3. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=repeat)
        rf.fit(X_train, y_train_skl)
        preds_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test_skl, preds_rf)
        results["Random Forest"].append(acc_rf)

        # 4. SVM (线性核)
        svm = SVC(kernel="linear", probability=False)
        svm.fit(X_train, y_train_skl)
        preds_svm = svm.predict(X_test)
        acc_svm = accuracy_score(y_test_skl, preds_svm)
        results["SVM"].append(acc_svm)

        # 5. AdaBoost (SKLearn 默认决策树桩)
        adb = AdaBoostClassifier(n_estimators=50, random_state=repeat)
        adb.fit(X_train, y_train_skl)
        preds_adb = adb.predict(X_test)
        acc_adb = accuracy_score(y_test_skl, preds_adb)
        results["AdaBoost (SKLearn)"].append(acc_adb)

        # ---------- QBoost ----------
        # 生成弱分类器并选前 M 个
        all_stumps = generate_weak_classifiers(X_train, num_thresholds=10)
        scores = np.abs([(clf.predict(X_train) * y_train).sum() for clf in all_stumps])
        top_idx = np.argsort(-np.array(scores))[:M]
        stumps = [all_stumps[i] for i in top_idx]

        # 训练 QBoost
        selected = qboost_train(stumps, X_train, y_train, lambda_reg=lambda_reg, num_reads=num_reads)
        # 测试集上预测
        preds_qb = predict_strong(selected, X_test)
        acc_qb = (preds_qb == y_test).mean()
        results["QBoost"].append(acc_qb)

        print(f"Repeat {repeat+1}/{n_repeats} 完成: "
              f"RandGuess={rg_mean:.3f}, LR={acc_lr:.3f}, DT={acc_dt:.3f}, RF={acc_rf:.3f}, "
              f"SVM={acc_svm:.3f}, AdaBoost={acc_adb:.3f}, QBoost={acc_qb:.3f}")

    # 计算平均与标准差
    summary = {}
    for algo, scores in results.items():
        scores_arr = np.array(scores)
        summary[algo] = (scores_arr.mean(), scores_arr.std())

    return summary


def main():
    classes = (4, 6)
    summary = evaluate_classifiers(classes=classes,
                                   test_size=0.2,
                                   n_repeats=30,
                                   M=200,
                                   lambda_reg=0.01,
                                   num_reads=50,
                                   random_guess_runs=1)
    print("\n===== 不同算法在二分类任务上的平均准确率（5 次重复） =====")
    for algo, (mean_acc, std_acc) in summary.items():
        print(f"{algo:20s} 平均准确率: {mean_acc*100:.2f}%  ± {std_acc*100:.2f}%")

    # 可选：画条形图进行可视化对比
    algos = list(summary.keys())
    means = [summary[a][0] for a in algos]
    stds = [summary[a][1] for a in algos]

    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(algos))
    plt.bar(bar_positions, means, yerr=stds, align='center', alpha=0.7, capsize=5)
    plt.xticks(bar_positions, algos, rotation=45, ha='right')
    plt.ylabel("Accuracy")
    plt.title(f"Comparison between Qboost and other algorithms. ({classes[0]} vs {classes[1]})")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
