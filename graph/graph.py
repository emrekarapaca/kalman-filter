import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "/Users/emre/Desktop/kalman/kalman_graph/pythonProject1/tracking_results.csv"

df = pd.read_csv(path)
mse = np.mean((df['Predicted X'] - df['Actual X'])**2 + (df['Predicted Y'] - df['Actual Y'])**2)

print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(10, 5))
plt.plot(df['Predicted X'], df['Predicted Y'], label='Predicted', marker='o')
plt.plot(df['Actual X'], df['Actual Y'], label='Actual', marker='x')
plt.title('Predicted vs Actual Trajectory')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid()
plt.show()
