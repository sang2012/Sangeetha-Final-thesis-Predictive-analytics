# ============================================
# Step2_clean
# ============================================
import pandas as pd
import numpy as np

# Load data
print("Loading data...")
df = pd.read_csv('era5_all_years.csv')
print(f"Rows: {len(df)}")

# Keep only useful columns
df = df[['time', 'latitude', 'longitude', 't2m', 'u10', 'v10', 'sp', 'swvl1', 'tp']]
print(f"Kept columns: {df.columns.tolist()}")

# Convert time
df['time'] = pd.to_datetime(df['time'])

# Check missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Convert units
df['t2m_celsius'] = df['t2m'] - 273.15       # Kelvin → Celsius
df['tp_mm'] = df['tp'] * 1000                 # metres → mm
df['sp_hpa'] = df['sp'] / 100                 # Pa → hPa

# Save
df.to_csv('era5_cleaned.csv', index=False)
print(f"\nSaved era5_cleaned.csv ({len(df)} rows)")
print(f"\nSample:")
print(df[['time', 'latitude', 'longitude', 't2m_celsius', 'tp_mm', 'sp_hpa', 'u10', 'v10', 'swvl1']].head(5))
