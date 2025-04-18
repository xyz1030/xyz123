import math

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps  # The maximum distance between two points for them to be considered neighbors
        self.min_samples = min_samples  # Minimum number of points required to form a dense region (cluster)
        self.labels = None  # Labels assigned to each point (cluster id or -1 for noise)

    def fit(self, X):
        n_samples = len(X)
        self.labels = [-1] * n_samples  # Initially, all apoints are considered noise (-1)
        visited = [False] * n_samples  # To keep track of visited points
        cluster_id = 0  # Start with the first cluster id

        for i in range(n_samples):
            if visited[i]:
                continue  # Skip if point is already visited
            visited[i] = True
            neighbors = self._region_query(X, i)  # Get all points within eps distance

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Mark as noise if not enough neighbors
            else:
                self._expand_cluster(X, i, neighbors, cluster_id, visited)  # Expand the cluster
                cluster_id += 1  # Increment the cluster id after completing one cluster

    def _expand_cluster(self, X, i, neighbors, cluster_id, visited):
        # Assign the cluster id to the current point
        self.labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            point = neighbors[j]
            if not visited[point]:
                visited[point] = True
                new_neighbors = self._region_query(X, point)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors  # Add the new neighbors to the list
            if self.labels[point] == -1:
                self.labels[point] = cluster_id  # Assign the point to the current cluster
            j += 1

    def _region_query(self, X, i):
        # Compute distances from point i to all other points and return neighbors within eps distance
        distances = [self._euclidean_distance(X[i], X[j]) for j in range(len(X))]
        return [idx for idx, dist in enumerate(distances) if dist <= self.eps]

    def _euclidean_distance(self, p1, p2):
        # Calculate Euclidean distance between points p1 and p2
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def predict(self):
        # Return the labels (cluster ids) for all the points
        return self.labels

def load_csv(filename):
    # Load CSV file and return data as a list of points
    X = []
    with open(filename, 'r') as f:
        for line in f:
            points = list(map(float, line.strip().split(',')))  # Convert each value to float
            X.append(points)
    return X

if __name__ == "__main__":
    # Load the data from the CSV file
    filename = "./CSV/dbscan.csv"  # Ensure the file exists and has the data
    X = load_csv(filename)

    # Create and fit the DBSCAN model
    model = DBSCAN(eps=1.0, min_samples=3)
    model.fit(X)

    # Get the cluster labels
    labels = model.predict()

    # Print the predicted labels for each point
    print("Predicted labels:", labels)
