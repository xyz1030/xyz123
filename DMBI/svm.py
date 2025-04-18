import csv
import random
import math

# --------- Load dataset manually from CSV ---------
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        data = []
        for row in reader:
            # Convert string values to float
            features = list(map(float, row[:-1]))
            label = int(row[-1])
            data.append((features, label))
    return data

# --------- Split dataset into training and testing ---------
def train_test_split(data, train_ratio=0.7):
    random.shuffle(data)
    split_index = int(train_ratio * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

# --------- Dot product of two vectors ---------
def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

# --------- Multiply vector by scalar ---------
def scalar_multiply(vector, scalar):
    return [x * scalar for x in vector]

# --------- Subtract two vectors ---------
def vector_subtract(v1, v2):
    return [x - y for x, y in zip(v1, v2)]

# --------- Add two vectors ---------
def vector_add(v1, v2):
    return [x + y for x, y in zip(v1, v2)]

# --------- Compute hinge loss ---------
def hinge_loss(data, weights, bias, lambda_param):
    loss = 0
    for features, label in data:
        y = 1 if label == 1 else -1
        margin = y * (dot(features, weights) + bias)
        loss += max(0, 1 - margin)
    weight_square = sum(w ** 2 for w in weights)
    return 0.5 * weight_square + lambda_param * loss

# --------- Predict function ---------
def predict(features, weights, bias):
    return 1 if (dot(features, weights) + bias) >= 0 else -1

# --------- Accuracy calculation ---------
def calculate_accuracy(data, weights, bias):
    correct = 0
    total = len(data)
    for features, label in data:
        true_y = 1 if label == 1 else -1
        pred = predict(features, weights, bias)
        if pred == true_y:
            correct += 1
    return correct / total

# --------- Load and prepare data ---------
data = load_csv('./CSV/svm_dataset.csv')

# --------- Preprocess labels to +1/-1 ---------
processed_data = []
for features, label in data:
    y = 1 if label == 1 else -1
    processed_data.append((features, y))

# --------- Split the dataset ---------
train_data, test_data = train_test_split(processed_data)

# --------- SVM Parameters ---------
n_features = len(train_data[0][0])
weights = [0.0] * n_features
bias = 0.0
learning_rate = 0.01
epochs = 1000
lambda_param = 0.01

# --------- Training with gradient descent ---------
for epoch in range(epochs):
    for features, label in train_data:
        margin = label * (dot(features, weights) + bias)

        if margin >= 1:
            # Only apply regularization
            gradient = scalar_multiply(weights, lambda_param)
            weights = vector_subtract(weights, scalar_multiply(gradient, learning_rate))
        else:
            # Apply regularization and hinge loss gradient
            gradient = vector_subtract(scalar_multiply(weights, lambda_param), scalar_multiply(features, label))
            weights = vector_subtract(weights, scalar_multiply(gradient, learning_rate))
            bias -= learning_rate * (-label)

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        loss = hinge_loss(train_data, weights, bias, lambda_param)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --------- Evaluation ---------
accuracy = calculate_accuracy(test_data, weights, bias)
print("Accuracy:", accuracy)

# --------- Print predictions ---------
print("Predictions vs True Labels:")
for features, true_label in test_data:
    pred = predict(features, weights, bias)
    print(f"Predicted: {pred}, Actual: {true_label}")
