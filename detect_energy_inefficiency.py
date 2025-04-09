
import numpy as np
import pandas as pd

# 1. Simulated data: occupancy, temperature, humidity, energy consumption
data = pd.DataFrame({
    'time': pd.date_range(start='2024-01-01 08:00', periods=10, freq='H'),
    'occupancy': [5, 10, 0, 25, 30, 0, 20, 15, 0, 10],
    'temperature': [22, 23, 24, 25, 26, 22, 23, 24, 22, 23],
    'humidity': [45, 50, 55, 60, 65, 50, 48, 47, 46, 50],
    'energy_consumption': [30, 60, 40, 90, 100, 45, 80, 70, 50, 65]
})

# 2. Define rule-based logic for inefficiency
def detect_inefficiency(row):
    if row['occupancy'] < 5 and row['energy_consumption'] > 40:
        return True
    return False

# 3. Apply the logic
data['inefficiency_detected'] = data.apply(detect_inefficiency, axis=1)

# 4. Print detected inefficiencies
print("Detected Inefficiencies:")
print(data[data['inefficiency_detected'] == True][['time', 'occupancy', 'energy_consumption']])

# Optional: visualize (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(data['time'], data['energy_consumption'], label='Energy Consumption')
    plt.scatter(data[data['inefficiency_detected']]['time'], 
                data[data['inefficiency_detected']]['energy_consumption'], 
                color='red', label='Inefficiency Detected')
    plt.xlabel('Time')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.title('Energy Usage and Detected Inefficiencies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
except ImportError:
    print("Install matplotlib to see visual output.")
