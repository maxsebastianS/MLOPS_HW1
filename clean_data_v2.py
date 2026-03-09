import pandas as pd

def clean_trip_data(df):
    df_clean = df[df['fare_amount'] >= 0].copy()
    df_clean = df_clean[df_clean['trip_distance'] >= 0]
    df_clean = df_clean[df_clean['trip_distance'] <= 50]
    df_clean = df_clean.dropna(subset=['payment_type', 'passenger_count', 'RatecodeID'])
    return df_clean

# Load and clean January 2021 data
df_jan = pd.read_parquet('green_tripdata_2021-01.parquet')
df_jan_clean = clean_trip_data(df_jan)
print(f"January 2021 cleaned rows: {df_jan_clean.shape[0]}")

# Load and clean February 2021 data
df_feb = pd.read_parquet('green_tripdata_2021-02.parquet')
df_feb_clean = clean_trip_data(df_feb)
print(f"February 2021 cleaned rows: {df_feb_clean.shape[0]}")

# Combine
df_combined = pd.concat([df_jan_clean, df_feb_clean], ignore_index=True)
print(f"Combined dataset rows: {df_combined.shape[0]}")

# Save
df_combined.to_parquet('green_tripdata_combined_cleaned.parquet', index=False)
print("Saved: green_tripdata_combined_cleaned.parquet")
