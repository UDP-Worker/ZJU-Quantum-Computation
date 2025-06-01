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

    accuracies = []
    n_samples = len(y_test)
    for _ in range(n_runs):

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
        X_train, X_test, y_train, y_test = load_data(classes=classes, test_size=test_size)

        y_train_skl = ((y_train + 1) // 2).astype(int)
        y_test_skl = ((y_test + 1) // 2).astype(int)

        rg_mean, rg_std = evaluate_random_guess(y_test, n_runs=random_guess_runs)
        results["Random Guess"].append(rg_mean)

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

        # 4. SVM
        svm = SVC(kernel="linear", probability=False)
        svm.fit(X_train, y_train_skl)
        preds_svm = svm.predict(X_test)
        acc_svm = accuracy_score(y_test_skl, preds_svm)
        results["SVM"].append(acc_svm)

        # 5. AdaBoost
        adb = AdaBoostClassifier(n_estimators=50, random_state=repeat)
        adb.fit(X_train, y_train_skl)
        preds_adb = adb.predict(X_test)
        acc_adb = accuracy_score(y_test_skl, preds_adb)
        results["AdaBoost (SKLearn)"].append(acc_adb)

        # 6. QBoost
        all_stumps = generate_weak_classifiers(X_train, num_thresholds=10)
        scores = np.abs([(clf.predict(X_train) * y_train).sum() for clf in all_stumps])
        top_idx = np.argsort(-np.array(scores))[:M]
        stumps = [all_stumps[i] for i in top_idx]

        selected = qboost_train(stumps, X_train, y_train, lambda_reg=lambda_reg, num_reads=num_reads)

        preds_qb = predict_strong(selected, X_test)
        acc_qb = (preds_qb == y_test).mean()
        results["QBoost"].append(acc_qb)

        print(f"Repeat {repeat+1}/{n_repeats} 完成: "
              f"RandGuess={rg_mean:.3f}, LR={acc_lr:.3f}, DT={acc_dt:.3f}, RF={acc_rf:.3f}, "
              f"SVM={acc_svm:.3f}, AdaBoost={acc_adb:.3f}, QBoost={acc_qb:.3f}")


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

    for algo, (mean_acc, std_acc) in summary.items():
        print(f"{algo:20s} Average accuracy: {mean_acc*100:.2f}%  ± {std_acc*100:.2f}%")

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
