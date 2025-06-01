import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class DecisionStump:

    def __init__(self, feature_index: int, threshold: float, polarity: int = 1):
        self.feature_index = feature_index
        self.threshold = threshold
        self.polarity = polarity

    def predict(self, X: np.ndarray) -> np.ndarray:
        feature_values = X[:, self.feature_index]
        preds = np.ones(len(feature_values), dtype=int)
        if self.polarity == 1:
            preds[feature_values < self.threshold] = -1
        else:
            preds[feature_values > self.threshold] = -1
        return preds


def load_data(classes=(0, 1), test_size: float = 0.2, random_state: int = 42):

    digits = load_digits()
    X = digits.data
    y = digits.target
    class_a, class_b = classes
    mask = (y == class_a) | (y == class_b)
    X = X[mask]
    y = y[mask]

    y = np.where(y == class_a, -1, 1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def generate_weak_classifiers(X: np.ndarray, num_thresholds: int = 10):

    stumps = []
    n_features = X.shape[1]
    for feature_idx in range(n_features):
        values = X[:, feature_idx]
        min_v, max_v = values.min(), values.max()
        thresholds = np.linspace(min_v, max_v, num_thresholds)
        for thresh in thresholds:
            stumps.append(DecisionStump(feature_idx, thresh, polarity=1))
            stumps.append(DecisionStump(feature_idx, thresh, polarity=-1))
    return stumps


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data(classes=(6, 9))
    classifiers = generate_weak_classifiers(X_train, num_thresholds=10)
    print(f"generated {len(classifiers)} weak classifiers")
