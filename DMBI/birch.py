# birch_manual.py
import math

class BirchManual:
    def __init__(self, threshold=0.5, branching_factor=50):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.clusters = []

    def fit(self, X):
        for point in X:
            found = False
            for cluster in self.clusters:
                dist = self._euclidean_distance(cluster['center'], point)
                if dist <= self.threshold:
                    cluster['points'].append(point)
                    cluster['center'] = self._mean(cluster['points'])
                    found = True
                    break
            if not found and len(self.clusters) < self.branching_factor:
                self.clusters.append({
                    'center': point,
                    'points': [point]
                })

    def predict(self, X):
        return [self._predict_point(p) for p in X]

    def _predict_point(self, x):
        min_dist = float('inf')
        closest_index = -1
        for i, cluster in enumerate(self.clusters):
            dist = self._euclidean_distance(cluster['center'], x)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        return closest_index

    def _euclidean_distance(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _mean(self, points):
        dim = len(points[0])
        total = [0.0] * dim
        for point in points:
            for i in range(dim):
                total[i] += point[i]
        return [total[i] / len(points) for i in range(dim)]

def load_csv(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() == "":
                continue
            parts = line.strip().split(',')
            point = [float(x) for x in parts]
            data.append(point)
    return data

# Sample usage
if __name__ == "__main__":
    filename = "./CSV/birch.csv"  # Ensure this file exists
    X = load_csv(filename)
    model = BirchManual(threshold=0.5, branching_factor=50)
    model.fit(X)
    labels = model.predict(X)

    print("Predicted labels:")
    for i, label in enumerate(labels):
        print(f"Point {i+1}: Cluster {label}")
