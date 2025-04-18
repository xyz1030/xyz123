import pandas as pd
from itertools import product

# Prior and Conditional Probability Tables
P_B = {True: 0.001, False: 0.999}
P_E = {True: 0.002, False: 0.998}
P_A = {
    (True, True): {True: 0.95, False: 0.05},
    (True, False): {True: 0.94, False: 0.06},
    (False, True): {True: 0.29, False: 0.71},
    (False, False): {True: 0.001, False: 0.999}
}
P_J = {True: {True: 0.90, False: 0.10}, False: {True: 0.05, False: 0.95}}
P_M = {True: {True: 0.70, False: 0.30}, False: {True: 0.01, False: 0.99}}

# Function to calculate joint probability
def joint_probability(b, e, a, j, m):
    return (P_B[b] *
            P_E[e] *
            P_A[(b, e)][a] *
            P_J[a][j] *
            P_M[a][m])

# Function to query the probability of Burglary given calls
def query_burglary(johnCalls=True, maryCalls=True):
    probs = {True: 0.0, False: 0.0}
    
    # Read data from CSV
    data = pd.read_csv('./CSV/bbn.csv')
    
    # Loop through all combinations of hidden variables B, E, and A from the CSV
    for _, row in data.iterrows():
        b = row['B']
        e = row['E']
        a = row['A']
        j = row['JohnCalls'] == 1
        m = row['MaryCalls'] == 1
        
        jp = joint_probability(b, e, a, j, m)
        probs[b] += jp

    # Normalize the result
    total = probs[True] + probs[False]
    if total > 0:  # To avoid division by zero
        probs[True] /= total
        probs[False] /= total
    return probs

# Run the query and print the result
result = query_burglary(johnCalls=True, maryCalls=True)
print(f"P(Burglary | JohnCalls=True, MaryCalls=True): {result[True]:.5f}")
