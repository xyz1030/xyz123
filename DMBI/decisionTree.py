# Without Graph
# manual_decision_tree.py
import csv

# ---- Configuration ----
FILENAME = './CSV/titanic.csv'       # CSV file with header: Age,Passenger Class,Sex,SibSp,Parch,Fare,Survived
MAX_DEPTH = 6

# Feature definitions
FEATURE_NAMES = [
    'Age',
    'Passenger Class',
    'Sex',
    'No of Siblings or Spouses on Board',
    'No of Parents or Children on Board',
    'Passenger Fare'
]
FEATURE_TYPES = ['num', 'cat', 'cat', 'num', 'num', 'num']

# Load CSV manually
def load_csv(filename):
    data = []
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            inputs = [
                float(row[0]),      # Age
                row[1].strip(),     # Passenger Class
                row[2].strip(),     # Sex
                float(row[3]),      # SibSp
                float(row[4]),      # Parch
                float(row[5])       # Fare
            ]
            label = row[6].strip()  # Survived: 'Yes' or 'No'
            data.append((inputs, label))
    return data

# Build categories for categorical features
def build_categories(data):
    cats = {}
    for idx, ftype in enumerate(FEATURE_TYPES):
        if ftype == 'cat':
            cats[idx] = set(sample[0][idx] for sample in data)
    return cats

# Manual label counting
def count_labels(labels):
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts

# Gini impurity calculation
def gini(labels):
    counts = count_labels(labels)
    total = len(labels)
    imp = 1.0
    for cnt in counts.values():
        p = cnt / total
        imp -= p * p
    return imp

# Find best split
def best_split(data):
    X = [sample[0] for sample in data]
    y = [sample[1] for sample in data]
    best = None
    best_gini = float('inf')
    n = len(y)
    for i, ftype in enumerate(FEATURE_TYPES):
        values = sorted(set(x[i] for x in X)) if ftype == 'num' else set(x[i] for x in X)
        for thr in values:
            if ftype == 'num':
                left = [y[j] for j in range(n) if X[j][i] <= thr]
                right = [y[j] for j in range(n) if X[j][i] > thr]
            else:
                left = [y[j] for j in range(n) if X[j][i] == thr]
                right = [y[j] for j in range(n) if X[j][i] != thr]
            if not left or not right:
                continue
            g = (len(left)/n)*gini(left) + (len(right)/n)*gini(right)
            if g < best_gini:
                best_gini = g
                best = (i, thr, ftype)
    return best

# Build decision tree recursively
def build_tree(data, depth=0, max_depth=MAX_DEPTH):
    labels = [sample[1] for sample in data]
    counts = count_labels(labels)
    # Stopping conditions
    if len(counts) == 1 or depth >= max_depth:
        pred = max(counts, key=lambda k: counts[k])
        return {'leaf': True, 'prediction': pred, 'counts': counts}
    split = best_split(data)
    if split is None:
        pred = max(counts, key=lambda k: counts[k])
        return {'leaf': True, 'prediction': pred, 'counts': counts}
    feat, thr, ftype = split
    # Partition data
    if ftype == 'num':
        left_data  = [s for s in data if s[0][feat] <= thr]
        right_data = [s for s in data if s[0][feat] > thr]
    else:
        left_data  = [s for s in data if s[0][feat] == thr]
        right_data = [s for s in data if s[0][feat] != thr]
    # Recurse
    left_tree  = build_tree(left_data,  depth+1, max_depth)
    right_tree = build_tree(right_data, depth+1, max_depth)
    return {
        'leaf': False,
        'feature': feat,
        'threshold': thr,
        'ftype': ftype,
        'left': left_tree,
        'right': right_tree
    }

# Predict a single sample
def predict_sample(x, node):
    if node['leaf']:
        return node['prediction']
    feat, thr, ftype = node['feature'], node['threshold'], node['ftype']
    # decide branch based on feature type
    if ftype == 'num':
        branch = 'left' if x[feat] <= thr else 'right'
    else:
        branch = 'left' if x[feat] == thr else 'right'
    return predict_sample(x, node[branch])

# Pretty-print the tree
def print_tree(node, categories, depth=0, condition=None):
    indent = '|   ' * depth
    if depth == 0:
        print('Tree')
    # Branch condition line
    if condition:
        if node['leaf']:
            counts = node['counts']
            yes = counts.get('Yes', 0)
            no  = counts.get('No',  0)
            print(f"{indent}{condition}: {node['prediction']} {{Yes={yes}, No={no}}}")
            return
        else:
            print(f"{indent}{condition}")
    # Leaf without condition
    if node['leaf']:
        counts = node['counts']
        yes = counts.get('Yes', 0)
        no  = counts.get('No',  0)
        print(f"{indent}Predict: {node['prediction']} {{Yes={yes}, No={no}}}")
        return
    # Internal node
    feat = node['feature']
    name = FEATURE_NAMES[feat]
    thr  = node['threshold']
    if node['ftype'] == 'num':
        print_tree(node['right'], categories, depth+1, f"{name} > {thr:.3f}")
        print_tree(node['left'],  categories, depth+1, f"{name} <= {thr:.3f}")
    else:
        print_tree(node['left'],  categories, depth+1, f"{name} = {thr}")
        others = categories[feat] - {thr}
        rest = others.pop() if others else '...'
        print_tree(node['right'], categories, depth+1, f"{name} != {rest}")

# ---- Main Execution ----
if __name__ == '__main__':
    data       = load_csv(FILENAME)
    categories = build_categories(data)
    tree       = build_tree(data)
    print_tree(tree, categories)
    sample = data[0][0]
    print('\nPrediction for first sample:')
    print(sample, '->', predict_sample(sample, tree))



# With Graph
# manual_decision_tree.py
# import csv
# import matplotlib.pyplot as plt

# # ---- Configuration ----
# FILENAME = './CSV/titanic.csv'       # CSV file with header: Age,Passenger Class,Sex,SibSp,Parch,Fare,Survived
# MAX_DEPTH = 6

# # Feature definitions
# FEATURE_NAMES = [
#     'Age',
#     'Passenger Class',
#     'Sex',
#     'No of Siblings or Spouses on Board',
#     'No of Parents or Children on Board',
#     'Passenger Fare'
# ]
# FEATURE_TYPES = ['num', 'cat', 'cat', 'num', 'num', 'num']

# # Load CSV manually
# def load_csv(filename):
#     data = []
#     with open(filename, 'r', newline='') as f:
#         reader = csv.reader(f)
#         next(reader)  # skip header
#         for row in reader:
#             inputs = [
#                 float(row[0]),      # Age
#                 row[1].strip(),     # Passenger Class
#                 row[2].strip(),     # Sex
#                 float(row[3]),      # SibSp
#                 float(row[4]),      # Parch
#                 float(row[5])       # Fare
#             ]
#             label = row[6].strip()  # Survived: 'Yes' or 'No'
#             data.append((inputs, label))
#     return data

# # Build categories for categorical features
# def build_categories(data):
#     cats = {}
#     for idx, ftype in enumerate(FEATURE_TYPES):
#         if ftype == 'cat':
#             cats[idx] = set(sample[0][idx] for sample in data)
#     return cats

# # Manual label counting
# def count_labels(labels):
#     counts = {}
#     for label in labels:
#         counts[label] = counts.get(label, 0) + 1
#     return counts

# # Gini impurity calculation
# def gini(labels):
#     counts = count_labels(labels)
#     total = len(labels)
#     imp = 1.0
#     for cnt in counts.values():
#         p = cnt / total
#         imp -= p * p
#     return imp

# # Find best split
# def best_split(data):
#     X = [sample[0] for sample in data]
#     y = [sample[1] for sample in data]
#     best = None
#     best_gini = float('inf')
#     n = len(y)
#     for i, ftype in enumerate(FEATURE_TYPES):
#         values = sorted(set(x[i] for x in X)) if ftype == 'num' else set(x[i] for x in X)
#         for thr in values:
#             if ftype == 'num':
#                 left = [y[j] for j in range(n) if X[j][i] <= thr]
#                 right = [y[j] for j in range(n) if X[j][i] > thr]
#             else:
#                 left = [y[j] for j in range(n) if X[j][i] == thr]
#                 right = [y[j] for j in range(n) if X[j][i] != thr]
#             if not left or not right:
#                 continue
#             g = (len(left)/n)*gini(left) + (len(right)/n)*gini(right)
#             if g < best_gini:
#                 best_gini = g
#                 best = (i, thr, ftype)
#     return best

# # Build decision tree recursively
# def build_tree(data, depth=0, max_depth=MAX_DEPTH):
#     labels = [sample[1] for sample in data]
#     counts = count_labels(labels)
#     # Stopping conditions
#     if len(counts) == 1 or depth >= max_depth:
#         pred = max(counts, key=lambda k: counts[k])
#         return {'leaf': True, 'prediction': pred, 'counts': counts}
#     split = best_split(data)
#     if split is None:
#         pred = max(counts, key=lambda k: counts[k])
#         return {'leaf': True, 'prediction': pred, 'counts': counts}
#     feat, thr, ftype = split
#     # Partition data
#     if ftype == 'num':
#         left_data  = [s for s in data if s[0][feat] <= thr]
#         right_data = [s for s in data if s[0][feat] > thr]
#     else:
#         left_data  = [s for s in data if s[0][feat] == thr]
#         right_data = [s for s in data if s[0][feat] != thr]
#     # Recurse
#     left_tree  = build_tree(left_data,  depth+1, max_depth)
#     right_tree = build_tree(right_data, depth+1, max_depth)
#     return {
#         'leaf': False,
#         'feature': feat,
#         'threshold': thr,
#         'ftype': ftype,
#         'left': left_tree,
#         'right': right_tree
#     }

# # Predict a single sample
# def predict_sample(x, node):
#     if node['leaf']:
#         return node['prediction']
#     feat, thr, ftype = node['feature'], node['threshold'], node['ftype']
#     if ftype == 'num':
#         branch = 'left' if x[feat] <= thr else 'right'
#     else:
#         branch = 'left' if x[feat] == thr else 'right'
#     return predict_sample(x, node[branch])

# # ------------ Tree Plotting Functions ------------
# # Assign positions to nodes for plotting
# def assign_positions(node, x=0.0, y=0.0, positions=None, level=1, x_offset=1.0):
#     if positions is None:
#         positions = {}
#     nid = id(node)
#     positions[nid] = (x, y)
#     if not node['leaf']:
#         assign_positions(node['left'],  x - x_offset/level, y-1, positions, level+1, x_offset)
#         assign_positions(node['right'], x + x_offset/level, y-1, positions, level+1, x_offset)
#     return positions

# # Plot tree using matplotlib
# def plot_tree(node, categories):
#     positions = assign_positions(node)
#     fig, ax = plt.subplots(figsize=(12, 8))

#     def draw(n):
#         nid = id(n)
#         x, y = positions[nid]
#         if n['leaf']:
#             cnts = n['counts']
#             txt = f"{n['prediction']} (Y={cnts.get('Yes',0)}, N={cnts.get('No',0)})"
#             ax.text(x, y, txt, ha='center', va='center', bbox=dict(boxstyle='round', fc='lightgray'))
#         else:
#             feat = n['feature']
#             name = FEATURE_NAMES[feat]
#             thr  = n['threshold']
#             label = f"{name}\n<= {thr:.3f}" if n['ftype']=='num' else f"{name} = {thr}"
#             ax.text(x, y, label, ha='center', va='center', bbox=dict(boxstyle='round', fc='white'))
#             for branch in ['left', 'right']:
#                 child = n[branch]
#                 xc, yc = positions[id(child)]
#                 ax.plot([x, xc], [y, yc], 'k-')
#                 draw(child)

#     draw(node)
#     ax.axis('off')
#     plt.show()

# # ---- Main Execution ----
# if __name__ == '__main__':
#     data       = load_csv(FILENAME)
#     categories = build_categories(data)
#     tree       = build_tree(data)

#     # Textual printout
#     def print_tree(node, categories, depth=0, condition=None):
#         indent = '|   ' * depth
#         if depth == 0:
#             print('Tree')
#         if condition:
#             if node['leaf']:
#                 cnts = node['counts']
#                 print(f"{indent}{condition}: {node['prediction']} {{Yes={cnts.get('Yes',0)}, No={cnts.get('No',0)}}}")
#                 return
#             else:
#                 print(f"{indent}{condition}")
#         if node['leaf'] and not condition:
#             cnts = node['counts']
#             print(f"{indent}Predict: {node['prediction']} {{Yes={cnts.get('Yes',0)}, No={cnts.get('No',0)}}}")
#             return
#         feat = node['feature']
#         name = FEATURE_NAMES[feat]
#         thr  = node['threshold']
#         if node['ftype']=='num':
#             print_tree(node['right'], categories, depth+1, f"{name} > {thr:.3f}")
#             print_tree(node['left'],  categories, depth+1, f"{name} <= {thr:.3f}")
#         else:
#             print_tree(node['left'],  categories, depth+1, f"{name} = {thr}")
#             others = categories[feat] - {thr}
#             rest = others.pop() if others else '...'
#             print_tree(node['right'], categories, depth+1, f"{name} != {rest}")

#     print_tree(tree, categories)
#     sample = data[0][0]
#     print('\nPrediction for first sample:')
#     print(sample, '->', predict_sample(sample, tree))

#     # Graphical display
#     plot_tree(tree, categories)
