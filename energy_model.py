
import csv
import numpy as np

# Load data from CSV (features: temp, occupancy, hour; target: energy)
def load_data(filename):
    X = []
    y = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            X.append([1.0, float(row[0]), float(row[1]), float(row[2])])  # add bias term
            y.append(float(row[3]))
    return np.array(X), np.array(y)

# Train using normal equation
def train_linear_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Predict
def predict(X, theta):
    return X @ theta

# Save predictions to file
def save_predictions(filename, predictions):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["Predicted Energy Consumption"])
        for value in predictions:
            writer.writerow([value])

# Example Usage
X, y = load_data('building_energy.csv')
theta = train_linear_regression(X, y)
predictions = predict(X, theta)
save_predictions('predicted_energy.csv', predictions)

print("Model coefficients:", theta)
print("Predictions saved to predicted_energy.csv")
