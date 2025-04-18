import math
import random

# Helper function to load CSV from scratch (without numpy)
def load_csv(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data = []
    labels = []

    # Skip the header row (if present)
    for line in lines[1:]:
        values = line.strip().split(',')
        data.append([float(v) for v in values[:-1]])
        labels.append(int(values[-1]))

    return data, labels

# Function to split the dataset (train-test split) without numpy
def train_test_split(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    indices = list(range(len(X)))
    random.shuffle(indices)
    test_size = int(len(X) * test_size)

    X_train = [X[i] for i in indices[:-test_size]]
    X_test = [X[i] for i in indices[-test_size:]]
    y_train = [y[i] for i in indices[:-test_size]]
    y_test = [y[i] for i in indices[-test_size:]]

    return X_train, X_test, y_train, y_test

# Decision Tree Classifier (Stump - max depth 1) without numpy
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.feature_index = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def _majority_class(self, y, mask, sample_weight=None):
        weighted_counts = {}
        for i, include in enumerate(mask):
            if include:
                label = y[i]
                weight = sample_weight[i] if sample_weight else 1
                weighted_counts[label] = weighted_counts.get(label, 0) + weight

        if not weighted_counts:
            return None
        return max(weighted_counts, key=weighted_counts.get)

    def fit(self, X, y, sample_weight=None):
        best_gini = float('inf')
        for feature_index in range(len(X[0])):
            thresholds = set(row[feature_index] for row in X)
            for threshold in thresholds:
                left_mask = [row[feature_index] <= threshold for row in X]
                right_mask = [not lm for lm in left_mask]

                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                left_class = self._majority_class(y, left_mask, sample_weight)
                right_class = self._majority_class(y, right_mask, sample_weight)

                if left_class is None or right_class is None:
                    continue

                gini = self._calculate_gini(y, left_mask, right_mask, sample_weight)

                if gini < best_gini:
                    best_gini = gini
                    self.feature_index = feature_index
                    self.threshold = threshold
                    self.left_class = left_class
                    self.right_class = right_class

    def _calculate_gini(self, y, left_mask, right_mask, sample_weight):
        left_size = sum(left_mask)
        right_size = sum(right_mask)
        total_size = len(y)

        left_impurity = self._gini_impurity(y, left_mask, sample_weight)
        right_impurity = self._gini_impurity(y, right_mask, sample_weight)

        gini = (left_size / total_size) * left_impurity + (right_size / total_size) * right_impurity
        return gini

    def _gini_impurity(self, y, mask, sample_weight=None):
        total_weight = sum(sample_weight[i] if sample_weight else 1 for i, include in enumerate(mask) if include)
        class_counts = {}

        for i, include in enumerate(mask):
            if include:
                label = y[i]
                weight = sample_weight[i] if sample_weight else 1
                class_counts[label] = class_counts.get(label, 0) + weight

        probs = [count / total_weight for count in class_counts.values()]
        return 1 - sum(p ** 2 for p in probs)

    def predict(self, X):
        predictions = []
        for row in X:
            if row[self.feature_index] <= self.threshold:
                predictions.append(self.left_class)
            else:
                predictions.append(self.right_class)
        return predictions

# AdaBoost from Scratch (without numpy)
def adaboost(X_train, y_train, n_estimators):
    classifiers = []
    alphas = []
    n = len(X_train)
    sample_weights = [1 / n] * n

    for _ in range(n_estimators):
        clf = DecisionTree(max_depth=1)
        clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = clf.predict(X_train)

        err = sum(sample_weights[i] for i in range(n) if y_pred[i] != y_train[i])
        err = max(err, 1e-10)  # Avoid division by zero

        alpha = 0.5 * math.log((1 - err) / err)

        # Update weights
        new_weights = []
        for i in range(n):
            weight = sample_weights[i] * math.exp(-alpha * y_train[i] * y_pred[i])
            new_weights.append(weight)

        total = sum(new_weights)
        sample_weights = [w / total for w in new_weights]

        classifiers.append(clf)
        alphas.append(alpha)

    return classifiers, alphas

# Predict function for AdaBoost (without numpy)
def predict(X, classifiers, alphas):
    predictions = [0] * len(X)
    for clf, alpha in zip(classifiers, alphas):
        y_pred = clf.predict(X)
        for i in range(len(X)):
            predictions[i] += alpha * y_pred[i]
    return [1 if p > 0 else -1 for p in predictions]

# Accuracy function from scratch (without numpy)
def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

# Confusion Matrix from scratch (without numpy)
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == -1 and pred == -1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == -1 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == -1)
    return [[tp, fp], [fn, tn]]

# Main Program
if __name__ == "__main__":
    X, y = load_csv('./CSV/adaboost_dataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert labels to -1 and 1
    y_train = [1 if label == 1 else -1 for label in y_train]
    y_test = [1 if label == 1 else -1 for label in y_test]

    classifiers, alphas = adaboost(X_train, y_train, n_estimators=10)
    y_pred = predict(X_test, classifiers, alphas)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:", conf_matrix)
