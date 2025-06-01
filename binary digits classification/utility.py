# utility.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class DecisionStump:
    """单像素阈值决策桩"""
    def __init__(self, feature_index: int, threshold: float, polarity: int = 1):
        self.feature_index = feature_index
        self.threshold = threshold
        self.polarity = polarity

    def predict(self, X: np.ndarray) -> np.ndarray:
        vals = X[:, self.feature_index]
        preds = np.ones(len(vals), dtype=int)
        if self.polarity == 1:
            preds[vals < self.threshold] = -1
        else:
            preds[vals > self.threshold] = -1
        return preds

class PixelDiffStump:
    """像素差分阈值桩：比较 feature_i - feature_j"""
    def __init__(self, i: int, j: int, threshold: float, polarity: int = 1):
        self.i, self.j = i, j
        self.threshold = threshold
        self.polarity = polarity

    def predict(self, X: np.ndarray) -> np.ndarray:
        vals = X[:, self.i] - X[:, self.j]
        preds = np.ones(len(vals), dtype=int)
        if self.polarity == 1:
            preds[vals < self.threshold] = -1
        else:
            preds[vals > self.threshold] = -1
        return preds

class PCAStump:
    """PCA 投影阈值桩"""
    def __init__(self, component_index: int, threshold: float, pca: PCA, polarity: int = 1):
        self.component_index = component_index
        self.threshold = threshold
        self.pca = pca
        self.polarity = polarity

    def predict(self, X: np.ndarray) -> np.ndarray:
        proj = self.pca.transform(X)[:, self.component_index]
        preds = np.ones(len(proj), dtype=int)
        if self.polarity == 1:
            preds[proj < self.threshold] = -1
        else:
            preds[proj > self.threshold] = -1
        return preds

def load_data(classes=(0, 1), test_size: float = 0.2, random_state: int = 42):
    digits = load_digits()
    X, y = digits.data, digits.target
    a, b = classes
    mask = (y == a) | (y == b)
    X, y = X[mask], y[mask]
    y = np.where(y == a, -1, 1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_stumps(X: np.ndarray, y: np.ndarray, stumps: list) -> np.ndarray:
    """
    计算每个桩在 (X, y) 上的准确率，返回 accuracies 数组
    """
    accuracies = np.zeros(len(stumps))
    for idx, stump in enumerate(stumps):
        preds = stump.predict(X)
        accuracies[idx] = np.mean(preds == y)
    return accuracies

def generate_weak_classifiers(
        X: np.ndarray,
        y: np.ndarray,
        num_stumps: int = 200,
        num_thresholds: int = 10,
        num_diff_pairs: int = 100,
        n_pca_components: int = 5,
        random_state: int = 42
) -> list:
    """
    生成多类型弱分类器，然后根据在 (X, y) 上的表现挑选前 num_stumps 个最优桩。
    但会先剔除那些在训练集上恒定输出的桩（只有 +1 或只有 -1）。
    """
    rng = np.random.RandomState(random_state)
    n_features = X.shape[1]
    candidates = []

    # 1. 单像素阈值桩
    for i in range(n_features):
        vals = X[:, i]
        threshs = np.linspace(vals.min(), vals.max(), num_thresholds)
        for t in threshs:
            candidates.append(DecisionStump(i, t, polarity=1))
            candidates.append(DecisionStump(i, t, polarity=-1))

    # 2. 像素差分阈值桩
    features = np.arange(n_features)
    pairs = [tuple(rng.choice(features, 2, replace=False)) for _ in range(num_diff_pairs)]
    for (i, j) in pairs:
        diffs = X[:, i] - X[:, j]
        threshs = np.linspace(diffs.min(), diffs.max(), num_thresholds)
        for t in threshs:
            candidates.append(PixelDiffStump(i, j, t, polarity=1))
            candidates.append(PixelDiffStump(i, j, t, polarity=-1))

    # 3. PCA 投影阈值桩
    pca = PCA(n_components=n_pca_components, random_state=random_state).fit(X)
    proj = pca.transform(X)
    for comp in range(n_pca_components):
        vals = proj[:, comp]
        threshs = np.linspace(vals.min(), vals.max(), num_thresholds)
        for t in threshs:
            candidates.append(PCAStump(comp, t, pca, polarity=1))
            candidates.append(PCAStump(comp, t, pca, polarity=-1))

    # —— 先剔除在训练集上恒定输出的桩 ——
    filtered = []
    for stump in candidates:
        preds = stump.predict(X)
        # 只有当桩在训练集上既有 +1 也有 -1 输出时，才保留
        if preds.min() < preds.max():
            filtered.append(stump)

    # 评估剩余桩的准确率
    accuracies = np.array([np.mean(stump.predict(X) == y) for stump in filtered])
    # 按准确率降序，取前 num_stumps 个
    top_idxs = np.argsort(accuracies)[::-1][:num_stumps]
    return [filtered[i] for i in top_idxs]


if __name__ == "__main__":
    # 简单测试：二分类 0 vs 3
    X_train, X_test, y_train, y_test = load_data(classes=(0, 3))
    stumps = generate_weak_classifiers(X_train, y_train, num_stumps=100)
    print(f"最终挑选了 {len(stumps)} 个弱分类器")
    # 查看前几个桩的准确率
    accs = evaluate_stumps(X_train, y_train, stumps)
    print("前5个桩的准确率：", accs[:5])
