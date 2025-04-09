
import numpy as np
import pandas as pd

# 1. Create sample dataset
data = pd.DataFrame({
    'occupancy': [5, 10, 20, 30, 15, 25, 35, 40, 45, 50],
    'temperature': [22, 23, 24, 25, 21, 22, 26, 27, 28, 29],
    'humidity': [45, 50, 55, 60, 48, 52, 65, 63, 67, 70],
    'energy': [30, 50, 65, 90, 45, 70, 95, 110, 120, 140]
})

# 2. Prepare feature matrix X and target vector y
X = data[['occupancy', 'temperature', 'humidity']].values
y = data['energy'].values.reshape(-1, 1)

# 3. Add bias term (column of 1s) to X
X_b = np.hstack([np.ones((X.shape[0], 1)), X])  # shape: (n_samples, n_features+1)

# 4. Calculate weights using Normal Equation: w = (XᵀX)⁻¹ Xᵀy
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 5. Make prediction with new data (occupancy=32, temperature=24, humidity=56)
new_data = np.array([[1, 32, 24, 56]])  # include bias term
predicted_energy = new_data.dot(theta_best)[0][0]

# 6. Display result
print("Model Coefficients:", theta_best.flatten())
print(f"Predicted Energy Consumption: {predicted_energy:.2f} kWh")
