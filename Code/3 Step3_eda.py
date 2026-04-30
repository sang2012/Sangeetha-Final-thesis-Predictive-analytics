# ============================================
# Step3_eda
# ============================================
import pandas as pd
import numpy as np

print("Loading cleaned data...")
df = pd.read_csv('era5_cleaned.csv', parse_dates=['time'])

# ============================================
# 1. Basic info
# ============================================
print(f"\nRows: {len(df)}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"Unique dates: {df['time'].dt.date.nunique()}")
print(f"Unique latitudes: {df['latitude'].nunique()}")
print(f"Unique longitudes: {df['longitude'].nunique()}")
print(f"Grid points: {df['latitude'].nunique()} x {df['longitude'].nunique()} = {df['latitude'].nunique() * df['longitude'].nunique()}")

# ============================================
# 2. Statistics for each variable
# ============================================
print("\n" + "="*50)
print("STATISTICS")
print("="*50)
print(df[['t2m_celsius', 'tp_mm', 'sp_hpa', 'u10', 'v10', 'swvl1']].describe().round(3))

# ============================================
# 3. Monthly averages
# ============================================
df['month'] = df['time'].dt.month
monthly = df.groupby('month')[['t2m_celsius', 'tp_mm', 'swvl1']].mean().round(3)
print("\n" + "="*50)
print("MONTHLY AVERAGES")
print("="*50)
print(monthly)

# ============================================
# 4. Check for extreme values
# ============================================
print("\n" + "="*50)
print("EXTREME VALUES")
print("="*50)
print(f"Max temperature: {df['t2m_celsius'].max():.2f} °C")
print(f"Min temperature: {df['t2m_celsius'].min():.2f} °C")
print(f"Max precipitation: {df['tp_mm'].max():.2f} mm")
print(f"Max wind speed: {np.sqrt(df['u10']**2 + df['v10']**2).max():.2f} m/s")

print("\nEDA complete!")