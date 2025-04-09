
import csv
import numpy as np

# Load dataset
def load_data(filename):
    X = []
    y = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            X.append([1.0, float(row[0]), float(row[1]), float(row[2]), float(row[3])])  # Bias + 4 features
            y.append(float(row[4]))
    return np.array(X), np.array(y)

# Linear Regression (Normal Equation)
def train(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Predict
def predict(X, theta):
    return X @ theta

# Analyze results for optimization suggestions
def analyze(predictions, threshold=300):
    tips = []
    for i, value in enumerate(predictions):
        if value > threshold:
            tips.append(f"Time index {i}: High energy ({value:.2f}). Suggest reducing HVAC or lighting.")
    return tips

# Save outputs
def save_analysis(filename, predictions, tips):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Prediction", "Suggestion"])
        for i, pred in enumerate(predictions):
            tip = tips[i] if i < len(tips) else ""
            writer.writerow([pred, tip])

# Main
X, y = load_data('nzeb_energy.csv')
theta = train(X, y)
preds = predict(X, theta)
tips = analyze(preds)
save_analysis('nzeb_analysis.csv', preds, tips)

print("Analysis complete. Check nzeb_analysis.csv")
