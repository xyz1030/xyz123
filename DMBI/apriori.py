# apriori_csv.py
import csv
from collections import defaultdict

class Apriori:
    def __init__(self, min_support=0.3, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []

    def fit(self, transactions):
        itemsets = self._generate_itemsets(transactions)
        itemsets = self._filter_itemsets(itemsets, transactions)
        while itemsets:
            self.frequent_itemsets.extend(itemsets)
            itemsets = self._generate_next_itemsets(itemsets, transactions)
            itemsets = self._filter_itemsets(itemsets, transactions)

    def _generate_itemsets(self, transactions):
        itemsets = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                itemsets[frozenset([item])] += 1
        return itemsets

    def _filter_itemsets(self, itemsets, transactions):
        min_count = len(transactions) * self.min_support
        return [itemset for itemset, count in itemsets.items() if count >= min_count]

    def _generate_next_itemsets(self, itemsets, transactions):
        next_itemsets = defaultdict(int)
        itemsets = list(itemsets)
        for i in range(len(itemsets)):
            for j in range(i+1, len(itemsets)):
                combined = itemsets[i].union(itemsets[j])
                if len(combined) == len(itemsets[i]) + 1:
                    for transaction in transactions:
                        if combined.issubset(transaction):
                            next_itemsets[combined] += 1
        return next_itemsets

    def predict(self):
        return self.frequent_itemsets

def load_csv(filename):
    transactions = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned_row = [item.strip() for item in row if item.strip() != '']
            transactions.append(set(cleaned_row))
    return transactions

if __name__ == "__main__":
    filename = "./CSV/apriori.csv"  # CSV in same directory
    transactions = load_csv(filename)
    model = Apriori(min_support=0.3)
    model.fit(transactions)
    frequent_itemsets = model.predict()
    print("Frequent itemsets:")
    for itemset in frequent_itemsets:
        print(set(itemset))
