import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load combined dataset (Jan + Feb 2021)
df = pd.read_parquet('green_tripdata_combined_cleaned.parquet')

X = df[['trip_distance', 'passenger_count', 'RatecodeID', 'extra']].copy()
y = df['fare_amount'].copy()

# Same split strategy as V1
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Combined dataset size: {df.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Load V1 model (trained on Jan 2021 data only)
model = joblib.load('regression_model.pkl')

# Evaluate on new test set
y_pred_test = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)

print("\n===== V2: V1 MODEL ON COMBINED TEST SET (Jan + Feb 2021) =====")
print(f"MAE:      {mae:.4f}")
print(f"RMSE:     {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

print("\n===== V1 REFERENCE (Jan 2021 test set) =====")
print("MAE:      2.6632")
print("RMSE:     7.1377")
print("R² Score: 0.7548")
