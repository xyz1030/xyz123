import math
import random

# Load CSV manually
def load_csv(filename):
    data = []
    with open(filename, 'r') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split(',')
            inputs = list(map(float, parts[:-1]))
            output = float(parts[-1])
            data.append((inputs, output))
    return data

# Normalize inputs manually (z-score)
def normalize_dataset(dataset):
    num_features = len(dataset[0][0])
    means = [0] * num_features
    std_devs = [0] * num_features
    for i in range(num_features):
        col = [sample[0][i] for sample in dataset]
        means[i] = sum(col) / len(col)
        std_devs[i] = math.sqrt(sum((x - means[i]) ** 2 for x in col) / len(col))
    # Normalize
    for i in range(len(dataset)):
        for j in range(num_features):
            dataset[i][0][j] = (dataset[i][0][j] - means[j]) / std_devs[j]
    return dataset

# Activation and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
def initialize_weights(n_inputs, n_hidden, n_outputs):
    weights_input_hidden = [[random.uniform(-1, 1) for _ in range(n_hidden)] for _ in range(n_inputs)]
    weights_hidden_output = [[random.uniform(-1, 1)] for _ in range(n_hidden)]
    return weights_input_hidden, weights_hidden_output

# Dot product
def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

# Forward pass
def forward(inputs, weights_input_hidden, weights_hidden_output):
    hidden_input = [dot(inputs, [weights_input_hidden[j][i] for j in range(len(inputs))]) for i in range(len(weights_input_hidden[0]))]
    hidden_output = [sigmoid(x) for x in hidden_input]
    final_input = [dot(hidden_output, [w[0] for w in weights_hidden_output])]
    final_output = [sigmoid(x) for x in final_input]
    return hidden_output, final_output

# Backward pass and weight update
def train(dataset, epochs, lr, n_inputs, n_hidden):
    weights_input_hidden, weights_hidden_output = initialize_weights(n_inputs, n_hidden, 1)
    error_history = []

    for epoch in range(epochs):
        total_error = 0
        for inputs, target in dataset:
            # Forward
            hidden_output, final_output = forward(inputs, weights_input_hidden, weights_hidden_output)
            output_error = target - final_output[0]
            total_error += abs(output_error)

            # Derivatives
            d_output = output_error * sigmoid_derivative(final_output[0])
            d_hidden = [d_output * weights_hidden_output[i][0] * sigmoid_derivative(hidden_output[i]) for i in range(n_hidden)]

            # Update output weights
            for i in range(n_hidden):
                weights_hidden_output[i][0] += lr * d_output * hidden_output[i]

            # Update input-hidden weights
            for i in range(n_inputs):
                for j in range(n_hidden):
                    weights_input_hidden[i][j] += lr * d_hidden[j] * inputs[i]

        avg_error = total_error / len(dataset)
        error_history.append(avg_error)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Mean Absolute Error: {avg_error:.4f}")

    return weights_input_hidden, weights_hidden_output, error_history

# Final Prediction
def predict(dataset, weights_input_hidden, weights_hidden_output):
    predictions = []
    for inputs, _ in dataset:
        _, final_output = forward(inputs, weights_input_hidden, weights_hidden_output)
        predictions.append(final_output[0])
    return predictions

# Main execution
filename = 'backProp.csv'
data = load_csv(filename)
data = normalize_dataset(data)

# Network setup
epochs = 5000
learning_rate = 0.1
n_inputs = 2
n_hidden = 10

weights_input_hidden, weights_hidden_output, error_history = train(data, epochs, learning_rate, n_inputs, n_hidden)

# Final predictions
print("\nFinal Predictions vs Actual Output:")
predictions = predict(data, weights_input_hidden, weights_hidden_output)
for i in range(len(data)):
    print(f"Input: {data[i][0]}, Actual: {data[i][1]}, Predicted: {round(predictions[i], 3)}")
