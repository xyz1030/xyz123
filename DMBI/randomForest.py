import csv
import random
from collections import Counter

# Load CSV manually
def load_csv(filename):
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        headers = next(lines)
        dataset = []
        for row in lines:
            dataset.append([float(row[0]), float(row[1]), int(row[2])])
        return dataset

# Split dataset manually
def train_test_split(data, test_size=0.3, seed=42):
    random.seed(seed)
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]

# Gini impurity calculation
def gini_impurity(groups, classes):
    total_samples = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        group_classes = [row[-1] for row in group]
        for class_val in classes:
            proportion = group_classes.count(class_val) / size
            score += proportion * proportion
        gini += (1 - score) * (size / total_samples)
    return gini

# Split dataset based on feature and threshold
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] <= value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# Find the best split
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = None, None, float('inf'), None
    for index in [0, 1]:  # 0: study_hours, 1: sleep_hours
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_impurity(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

# Create terminal node
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return Counter(outcomes).most_common(1)[0][0]

# Recursive split of tree
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])

    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Predict with a decision tree
def predict(node, row):
    if row[node['index']] <= node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        return node['right']

# Create a random subsample with replacement
def subsample(dataset, ratio):
    sample = []
    while len(sample) < int(len(dataset) * ratio):
        index = random.randrange(len(dataset))
        sample.append(dataset[index])
    return sample

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_ratio, n_trees):
    trees = []
    for _ in range(n_trees):
        sample = subsample(train, sample_ratio)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)

    predictions = []
    for row in test:
        tree_preds = [predict(tree, row) for tree in trees]
        final_pred = Counter(tree_preds).most_common(1)[0][0]
        predictions.append(final_pred)
    return predictions

# Accuracy Calculation
def accuracy_metric(actual, predicted):
    correct = sum([1 for i in range(len(actual)) if actual[i] == predicted[i]])
    return correct / len(actual)

# Confusion Matrix
def confusion_matrix(actual, predicted):
    cm = [[0, 0], [0, 0]]
    for a, p in zip(actual, predicted):
        cm[a][p] += 1
    return cm

# Main
filename = './CSV/RandomForest.csv'
dataset = load_csv(filename)
train_data, test_data = train_test_split(dataset, test_size=0.3)

# Actual and predicted labels
actual = [row[-1] for row in test_data]
predicted = random_forest(train_data, test_data, max_depth=3, min_size=1, sample_ratio=1.0, n_trees=10)

# Results
print("Accuracy:", accuracy_metric(actual, predicted))
print("Confusion Matrix:")
cm = confusion_matrix(actual, predicted)
for row in cm:
    print(row)
