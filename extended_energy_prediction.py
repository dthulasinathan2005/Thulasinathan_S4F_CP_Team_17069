
import numpy as np
import pandas as pd

# Sample extended dataset
data = pd.DataFrame({
    'occupancy': [5, 10, 0, 30, 15, 0, 20, 35, 45, 50],
    'temperature': [22, 23, 24, 25, 21, 22, 26, 27, 28, 29],
    'humidity': [45, 50, 55, 60, 48, 52, 65, 63, 67, 70],
    'light_level': [200, 400, 100, 800, 300, 120, 600, 700, 850, 900],
    'co2_level': [400, 450, 300, 600, 350, 320, 500, 550, 580, 620],
    'hvac_on': [1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    'hour': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'day_of_week': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 = Monday
    'energy': [30, 60, 35, 90, 45, 40, 85, 100, 120, 140]
})

# Create lag feature: previous hour's energy consumption
data['prev_energy'] = data['energy'].shift(1).fillna(method='bfill')

# Features to use
features = ['occupancy', 'temperature', 'humidity', 'light_level', 'co2_level',
            'hvac_on', 'hour', 'day_of_week', 'prev_energy']

X = data[features].values
y = data['energy'].values.reshape(-1, 1)

# Add bias term to X
X_b = np.hstack([np.ones((X.shape[0], 1)), X])

# Normal equation: theta = (XᵀX)^-1 Xᵀy
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Predict on new data sample (same order of features + bias)
new_sample = np.array([[1, 32, 24, 56, 700, 580, 1, 14, 2, 80]])  # 1 is for bias
predicted_energy = new_sample @ theta

print("Model Coefficients:", theta.flatten())
print(f"Predicted Energy Consumption: {predicted_energy[0][0]:.2f} kWh")
