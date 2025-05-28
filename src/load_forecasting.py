import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os

# Define the path to your dataset
data_path = os.path.join('..', 'data', 'smart_grid_load.csv')

try:
    # Load the dataset
    data = pd.read_csv(data_path, parse_dates=['Datetime'], index_col='Datetime')
    print("Data loaded successfully.")
except Exception as e:
    print("Error loading data:", e)
    exit()

# Ensure proper frequency
data = data.asfreq('h')  # lowercase 'h' to avoid deprecation warning
data = data.ffill()  # forward-fill missing values
print("Missing values handled.")

# STEP 1: Add lag features
data['Load_lag1'] = data['Load'].shift(1)
data['Load_lag2'] = data['Load'].shift(2)
data['Load_lag24'] = data['Load'].shift(24)

# STEP 2: Add time-based features
data['hour'] = data.index.hour
data['dayofweek'] = data.index.dayofweek

# Drop rows with NaN values (due to lagging)
data = data.dropna()
print("Feature engineering completed.")

# Define features and target
feature_cols = ['Load_lag1', 'Load_lag2', 'Load_lag24', 'hour', 'dayofweek']
X = data[feature_cols]
y = data['Load']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("Train-test split done.")

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed.")

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Load')
plt.title('Load Forecasting: Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.show()
