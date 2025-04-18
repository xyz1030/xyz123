import csv
import os
from collections import  Counter

class FPNode:
    def __init__(self, item, parent):
        self.item = item
        self.count = 1
        self.parent = parent
        self.children = {}
        self.link = None

    def increment(self):
        self.count += 1

class FPTree:
    def __init__(self):
        self.root = FPNode(None, None)
        self.header_table = {}

    def add_transaction(self, transaction, item_order):
        current_node = self.root
        for item in transaction:
            if item in current_node.children:
                current_node.children[item].increment()
            else:
                new_node = FPNode(item, current_node)
                current_node.children[item] = new_node
                # Update header table
                if item in self.header_table:
                    current_link = self.header_table[item]
                    while current_link.link:
                        current_link = current_link.link
                    current_link.link = new_node
                else:
                    self.header_table[item] = new_node
            current_node = current_node.children[item]

    def is_single_path(self, node=None):
        if node is None:
            node = self.root
        while node:
            if len(node.children) > 1:
                return False
            node = next(iter(node.children.values()), None)
        return True

class FPGrowth:
    def __init__(self, min_support=0.3):
        self.min_support = min_support
        self.frequent_itemsets = []

    def fit(self, transactions):
        item_counter = Counter()
        for transaction in transactions:
            item_counter.update(transaction)

        min_count = int(len(transactions) * self.min_support)
        frequent_items = {item for item, count in item_counter.items() if count >= min_count}
        item_order = {item: count for item, count in item_counter.items() if item in frequent_items}
        item_order = dict(sorted(item_order.items(), key=lambda x: (-x[1], x[0])))

        tree = FPTree()
        for transaction in transactions:
            sorted_items = [item for item in sorted(transaction, key=lambda x: item_order.get(x, 0), reverse=True)
                            if item in frequent_items]
            if sorted_items:
                tree.add_transaction(sorted_items, item_order)

        self._mine_tree(tree, [], min_count)

    def _mine_tree(self, tree, suffix, min_count):
        for item in sorted(tree.header_table, key=lambda x: x):
            new_suffix = suffix + [item]
            self.frequent_itemsets.append(frozenset(new_suffix))

            conditional_patterns = []
            node = tree.header_table[item]
            while node:
                path = []
                parent = node.parent
                while parent and parent.item:
                    path.append(parent.item)
                    parent = parent.parent
                if path:
                    conditional_patterns.append((list(reversed(path)), node.count))
                node = node.link

            conditional_tree = FPTree()
            conditional_counter = Counter()
            for path, count in conditional_patterns:
                for item_in_path in path:
                    conditional_counter[item_in_path] += count

            frequent_cond_items = {item for item, count in conditional_counter.items() if count >= min_count}
            item_order = dict(sorted({item: conditional_counter[item] for item in frequent_cond_items}.items(), key=lambda x: (-x[1], x[0])))

            for path, count in conditional_patterns:
                filtered = [item for item in path if item in frequent_cond_items]
                sorted_path = sorted(filtered, key=lambda x: item_order[x])
                if sorted_path:
                    current_node = conditional_tree.root
                    for item_in_path in sorted_path:
                        if item_in_path in current_node.children:
                            current_node.children[item_in_path].count += count
                        else:
                            new_node = FPNode(item_in_path, current_node)
                            new_node.count = count
                            current_node.children[item_in_path] = new_node
                            # header link
                            if item_in_path in conditional_tree.header_table:
                                link = conditional_tree.header_table[item_in_path]
                                while link.link:
                                    link = link.link
                                link.link = new_node
                            else:
                                conditional_tree.header_table[item_in_path] = new_node
                        current_node = current_node.children[item_in_path]

            if conditional_tree.root.children:
                self._mine_tree(conditional_tree, new_suffix, min_count)

    def predict(self):
        return self.frequent_itemsets

def load_csv(filename):
    transactions = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            transactions.append(set(row))
    return transactions

if __name__ == "__main__":
    filename = "./CSV/fpgrowth_data.csv"
    if not os.path.exists(filename):
        print("CSV file not found in current directory.")
    else:
        transactions = load_csv(filename)
        model = FPGrowth(min_support=0.3)  # You can adjust support
        model.fit(transactions)
        frequent_itemsets = model.predict()
        print("Frequent itemsets:")
        for itemset in sorted(frequent_itemsets, key=lambda x: (-len(x), x)):
            print(set(itemset))
