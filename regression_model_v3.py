import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load combined dataset (Jan + Feb 2021)
df = pd.read_parquet('green_tripdata_combined_cleaned.parquet')

X = df[['trip_distance', 'passenger_count', 'RatecodeID', 'extra']].copy()
y = df['fare_amount'].copy()

# Same split strategy
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Train set: {X_train.shape[0]}")
print(f"Validation set: {X_val.shape[0]}")
print(f"Test set: {X_test.shape[0]}")

# Train on combined data
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_val   = model.predict(X_val)
y_pred_test  = model.predict(X_test)

print("\n===== TRAINING SET =====")
print(f"MAE:      {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"RMSE:     {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"R² Score: {r2_score(y_train, y_pred_train):.4f}")

print("\n===== VALIDATION SET =====")
print(f"MAE:      {mean_absolute_error(y_val, y_pred_val):.4f}")
print(f"RMSE:     {np.sqrt(mean_squared_error(y_val, y_pred_val)):.4f}")
print(f"R² Score: {r2_score(y_val, y_pred_val):.4f}")

print("\n===== TEST SET =====")
mae_test  = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test   = r2_score(y_test, y_pred_test)
print(f"MAE:      {mae_test:.4f}")
print(f"RMSE:     {rmse_test:.4f}")
print(f"R² Score: {r2_test:.4f}")

print("\n===== MODEL COEFFICIENTS =====")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.6f}")
print(f"Intercept: {model.intercept_:.6f}")

joblib.dump(model, 'regression_model_v3.pkl')
print("\nModel saved to regression_model_v3.pkl")
