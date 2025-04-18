import csv

# Load dataset manually
def load_csv(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # first row is header
        for row in reader:
            data.append(dict(zip(headers, row)))
    return headers, data

# Calculate prior probabilities
def calculate_prior(data, target):
    counts = {}
    total = len(data)
    for row in data:
        label = row[target]
        counts[label] = counts.get(label, 0) + 1
    prior = {}
    for label in counts:
        prior[label] = counts[label] / total
    return prior

# Calculate likelihoods P(feature=value | target)
def calculate_likelihood(data, features, target):
    likelihood = {}
    target_values = set(row[target] for row in data)

    for feature in features:
        likelihood[feature] = {}

        # Get all values of this feature
        feature_values = set(row[feature] for row in data)

        for value in feature_values:
            likelihood[feature][value] = {}

            for t_val in target_values:
                num = 0
                denom = 0
                for row in data:
                    if row[target] == t_val:
                        denom += 1
                        if row[feature] == value:
                            num += 1
                likelihood[feature][value][t_val] = num / denom if denom != 0 else 0
    return likelihood

# Predict a new instance
def predict(instance, prior, likelihood, target_values):
    probs = {}
    for t_val in target_values:
        prob = prior.get(t_val, 0)
        for feature, val in instance.items():
            prob *= likelihood[feature].get(val, {}).get(t_val, 0)
        probs[t_val] = prob

    # Choose the class with the max probability
    return max(probs, key=probs.get)

# Main logic
filename = './CSV/basyen.csv'  # Your CSV should exist in same folder
headers, data = load_csv(filename)

target = 'Play'
features = [col for col in headers if col != target]

prior = calculate_prior(data, target)
likelihood = calculate_likelihood(data, features, target)
target_values = set(row[target] for row in data)

# Test instance (manually created)
test_instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Windy': 'True'}
prediction = predict(test_instance, prior, likelihood, target_values)

print("Prediction:", prediction)
