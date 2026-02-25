import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample hourly energy data (dummy data)
hours = np.arange(1, 25).reshape(-1, 1)
energy_usage = np.array([
    120, 115, 110, 105, 100, 130, 180, 200, 220, 250, 300, 320,
    310, 290, 270, 260, 240, 230, 210, 190, 170, 160, 140, 130
])

# Train model
model = LinearRegression()
model.fit(hours, energy_usage)

# Predict next 24 hours
future_hours = np.arange(25, 49).reshape(-1, 1)
predictions = model.predict(future_hours)

print("Predicted Energy for Next 24 Hours:")
print(predictions)

# Plot
plt.plot(hours, energy_usage, label="Current Usage")
plt.plot(future_hours, predictions, label="Predicted Usage")
plt.legend()
plt.xlabel("Hour")
plt.ylabel("Energy Usage")
plt.title("Energy Prediction Model")
plt.show()
