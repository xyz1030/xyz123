import csv

class CLIQUE:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.grid_clusters = {}

    def fit(self, X):
        # Iterating through each data point
        for i in range(len(X)):
            # Convert each data point into grid cell coordinates
            grid_cell = tuple(int(val // self.grid_size) for val in X[i])
            
            # Add the point to the corresponding grid cell
            if grid_cell not in self.grid_clusters:
                self.grid_clusters[grid_cell] = []
            self.grid_clusters[grid_cell].append(i)

    def predict(self, X):
        # Initialize the label list with -1 (indicating noise)
        labels = [-1] * len(X)
        cluster_id = 0
        
        # Assign a unique cluster ID to each grid cell
        for cell, indices in self.grid_clusters.items():
            for idx in indices:
                labels[idx] = cluster_id
            cluster_id += 1
        
        return labels

def load_csv(filename):
    X = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Convert each value to float and add to X
            X.append([float(val) for val in row])
    return X

if __name__ == "__main__":
    # Load dataset from CSV
    filename = "./CSV/clique.csv"  # Ensure this file exists in the same directory
    X = load_csv(filename)

    # Initialize the CLIQUE model with grid size of 5
    model = CLIQUE(grid_size=5)

    # Fit the model to the data
    model.fit(X)

    # Get the predicted labels for the clusters
    labels = model.predict(X)

    # Print the predicted cluster labels
    print("Predicted cluster labels:", labels)
