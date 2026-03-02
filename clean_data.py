import pandas as pd
import numpy as np

# Load the data
df = pd.read_parquet('green_tripdata_2021-01.parquet')

# Step 1: Remove rows with negative fares
df_clean = df[df['fare_amount'] >= 0].copy()
print(f"2. After removing negative fares: {df_clean.shape[0]} rows (removed {df.shape[0] - df_clean.shape[0]})")

# Step 2: Remove rows with negative distances
df_clean = df_clean[df_clean['trip_distance'] >= 0]
print(f"3. After removing negative distances: {df_clean.shape[0]} rows (removed {df.shape[0] - df_clean.shape[0]} total)")

# Step 3: Remove rows with trip_distance > 50
df_clean = df_clean[df_clean['trip_distance'] <= 50]
print(f"4. After filtering distance <= 50: {df_clean.shape[0]} rows (removed)")

# Step 4: Remove rows with missing values in key columns
df_clean = df_clean.dropna(subset=['payment_type', 'passenger_count', 'RatecodeID'])
print(f"5. After removing rows with missing values: {df_clean.shape[0]} rows")

# Save cleaned data
df_clean.to_parquet('green_tripdata_2021-01_cleaned.parquet', index=False)
