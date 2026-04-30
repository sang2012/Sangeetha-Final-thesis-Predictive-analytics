# ============================================
# Step_features
# ============================================
import pandas as pd
import numpy as np

print("Loading cleaned data...")
df = pd.read_csv('era5_cleaned.csv', parse_dates=['time'])
df = df.sort_values(['latitude', 'longitude', 'time']).reset_index(drop=True)

print(f"Rows: {len(df)}")

# ============================================
# 1. Temporal features
# ============================================
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day_of_year'] = df['time'].dt.dayofyear
df['season'] = df['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})
print("✅ Temporal features added")

# ============================================
# 2. Wind speed
# ============================================
df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
print("✅ Wind speed added")

# ============================================
# 3. Lag features (per grid point)
# ============================================
print("Creating lag features (this may take a few minutes)...")

lag_vars = ['t2m_celsius', 'tp_mm', 'swvl1']
lag_days = [1, 7, 30]

for var in lag_vars:
    for lag in lag_days:
        col_name = f'{var}_lag{lag}'
        df[col_name] = df.groupby(['latitude', 'longitude'])[var].shift(lag)

print("✅ Lag features added (1, 7, 30 days)")

# ============================================
# 4. Rolling statistics (per grid point)
# ============================================
print("Creating rolling features (this may take a few minutes)...")

for var in lag_vars:
    grouped = df.groupby(['latitude', 'longitude'])[var]
    df[f'{var}_roll7_mean'] = grouped.transform(lambda x: x.rolling(7, min_periods=1).mean())
    df[f'{var}_roll7_std'] = grouped.transform(lambda x: x.rolling(7, min_periods=1).std())
    df[f'{var}_roll30_mean'] = grouped.transform(lambda x: x.rolling(30, min_periods=1).mean())

print("✅ Rolling features added (7-day and 30-day)")

# ============================================
# 5. Anomaly variables
# ============================================
climatology = df.groupby(['latitude', 'longitude', 'month'])[lag_vars].transform('mean')
df['t2m_anomaly'] = df['t2m_celsius'] - climatology['t2m_celsius']
df['tp_anomaly'] = df['tp_mm'] - climatology['tp_mm']
df['swvl1_anomaly'] = df['swvl1'] - climatology['swvl1']
print("✅ Anomaly variables added")

# ============================================
# 6. Drop rows with NaN (from lags)
# ============================================
before = len(df)
df = df.dropna()
after = len(df)
print(f"\nDropped {before - after} rows with NaN (lag edges)")

# ============================================
# 7. Save
# ============================================
df.to_csv('era5_features.csv', index=False)
print(f"\nSaved era5_features.csv")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"\nAll columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")
