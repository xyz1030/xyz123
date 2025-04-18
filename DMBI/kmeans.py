# # with graph
# import csv
# import random
# import turtle
# import math

# # Step 1: Load Data from CSV
# def load_csv(filename):
#     data = []
#     with open(filename, 'r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             f1 = float(row['feature1'])
#             f2 = float(row['feature2'])
#             data.append([f1, f2])
#     return data

# # Step 2: Min-Max Scaling (Standardize)
# def scale_features(data):
#     f1_vals = [row[0] for row in data]
#     f2_vals = [row[1] for row in data]
#     min_f1, max_f1 = min(f1_vals), max(f1_vals)
#     min_f2, max_f2 = min(f2_vals), max(f2_vals)

#     scaled = []
#     for row in data:
#         scaled_f1 = (row[0] - min_f1) / (max_f1 - min_f1)
#         scaled_f2 = (row[1] - min_f2) / (max_f2 - min_f2)
#         scaled.append([scaled_f1, scaled_f2])
#     return scaled

# # Step 3: Euclidean Distance
# def distance(p1, p2):
#     return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# # Step 4: Initialize Centroids Randomly
# def initialize_centroids(data, k):
#     return random.sample(data, k)

# # Step 5: Assign Points to Nearest Cluster
# def assign_clusters(data, centroids):
#     clusters = [[] for _ in centroids]
#     for point in data:
#         distances = [distance(point, centroid) for centroid in centroids]
#         min_index = distances.index(min(distances))
#         clusters[min_index].append(point)
#     return clusters

# # Step 6: Update Centroids
# def update_centroids(clusters):
#     new_centroids = []
#     for cluster in clusters:
#         if cluster:
#             x = sum(p[0] for p in cluster) / len(cluster)
#             y = sum(p[1] for p in cluster) / len(cluster)
#             new_centroids.append([x, y])
#         else:
#             new_centroids.append([0, 0])  # If a cluster is empty
#     return new_centroids

# # Step 7: Check Convergence
# def has_converged(old_centroids, new_centroids):
#     return all(distance(old, new) < 1e-4 for old, new in zip(old_centroids, new_centroids))

# # Step 8: Full KMeans Algorithm
# def kmeans(data, k=3, max_iterations=100):
#     centroids = initialize_centroids(data, k)
#     for _ in range(max_iterations):
#         clusters = assign_clusters(data, centroids)
#         new_centroids = update_centroids(clusters)
#         if has_converged(centroids, new_centroids):
#             break
#         centroids = new_centroids
#     return clusters, centroids

# # Step 9: Visualize using turtle
# def draw_clusters(clusters, centroids):
#     screen = turtle.Screen()
#     screen.title("K-Means Clustering")
#     t = turtle.Turtle()
#     t.speed(0)
#     t.penup()

#     colors = ['blue', 'green', 'orange', 'purple', 'cyan']

#     # Draw points
#     for i, cluster in enumerate(clusters):
#         t.color(colors[i % len(colors)])
#         for point in cluster:
#             x = point[0] * 400 - 200
#             y = point[1] * 400 - 200
#             t.goto(x, y)
#             t.dot(10)

#     # Draw centroids
#     for centroid in centroids:
#         x = centroid[0] * 400 - 200
#         y = centroid[1] * 400 - 200
#         t.goto(x, y)
#         t.dot(20, "red")

#     screen.mainloop()

# # Step 10: Run the full flow
# if __name__ == "__main__":
#     filename = "./CSV/kmeans.csv"  # Your CSV with 'feature1' and 'feature2'
#     raw_data = load_csv(filename)
#     scaled_data = scale_features(raw_data)
#     clusters, centroids = kmeans(scaled_data, k=3)
    
#     print("Centroids:")
#     for c in centroids:
#         print([round(x, 3) for x in c])
    
#     draw_clusters(clusters, centroids)

# without graph
import csv
import random
import math

# Load data from CSV (assuming header: feature1,feature2)
def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # skip header
        for row in reader:
            data.append([float(row[0]), float(row[1])])
    return data

# Min-max scaling (standardization)
def scale_data(data):
    feature1 = [row[0] for row in data]
    feature2 = [row[1] for row in data]
    min_f1, max_f1 = min(feature1), max(feature1)
    min_f2, max_f2 = min(feature2), max(feature2)

    scaled = []
    for row in data:
        x = (row[0] - min_f1) / (max_f1 - min_f1)
        y = (row[1] - min_f2) / (max_f2 - min_f2)
        scaled.append([x, y])
    return scaled

# Euclidean distance
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Assign points to nearest centroid
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        dists = [distance(point, c) for c in centroids]
        cluster = dists.index(min(dists))
        clusters.append(cluster)
    return clusters

# Update centroids
def compute_centroids(data, clusters, k):
    sums = [[0, 0] for _ in range(k)]
    counts = [0] * k

    for i in range(len(data)):
        c = clusters[i]
        sums[c][0] += data[i][0]
        sums[c][1] += data[i][1]
        counts[c] += 1

    new_centroids = []
    for i in range(k):
        if counts[i] == 0:
            new_centroids.append([0, 0])  # avoid division by zero
        else:
            new_centroids.append([sums[i][0] / counts[i], sums[i][1] / counts[i]])
    return new_centroids

# Check convergence
def has_converged(old, new):
    for i in range(len(old)):
        if distance(old[i], new[i]) > 1e-4:
            return False
    return True

# Main KMeans function
def kmeans(data, k, max_iter=100):
    centroids = random.sample(data, k)
    for _ in range(max_iter):
        clusters = assign_clusters(data, centroids)
        new_centroids = compute_centroids(data, clusters, k)
        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Print clustered data
def print_result(data, clusters, centroids):
    print("Final Centroids:")
    for i, c in enumerate(centroids):
        print(f"Cluster {i}: ({round(c[0], 3)}, {round(c[1], 3)})")

    print("\nData points with cluster labels:")
    for i in range(len(data)):
        print(f"Point {i+1}: ({round(data[i][0], 3)}, {round(data[i][1], 3)}) → Cluster {clusters[i]}")

# Run the program
if __name__ == "__main__":
    filename = "./CSV/kmeans.csv"  # CSV file with feature1, feature2
    raw_data = load_data(filename)
    scaled_data = scale_data(raw_data)
    k = 3  # Number of clusters
    clusters, centroids = kmeans(scaled_data, k)
    print_result(scaled_data, clusters, centroids)
