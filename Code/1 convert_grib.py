# ============================================
# convert_grib python code
# ============================================
import xarray as xr
import pandas as pd

# ============================================
# STEP 1: Load main variables
# ============================================
print("Loading main variables...")
ds1 = xr.open_dataset("main_variables.grib", engine="cfgrib")
print(ds1)
print()

# ============================================
# STEP 2: Load precipitation (with all keys)
# ============================================
print("Loading precipitation...")
ds2 = xr.open_dataset(
    "precipitation.grib",
    engine="cfgrib",
    backend_kwargs={"indexpath": ""}
)
print(ds2)
print()

# ============================================
# STEP 3: Convert main variables
# ============================================
print("Converting main variables to dataframe...")
df1 = ds1.to_dataframe().reset_index()
print(f"Main variables: {len(df1)} rows")
print(f"Columns: {df1.columns.tolist()}")
print()

# ============================================
# STEP 4: Convert precipitation
# ============================================
print("Converting precipitation to dataframe...")
df2 = ds2.to_dataframe().reset_index()
print(f"Precipitation: {len(df2)} rows")
print(f"Columns: {df2.columns.tolist()}")
print()

# Check what time columns exist in precipitation
print("Precipitation time info:")
for col in df2.columns:
    print(f"  {col}: {df2[col].dtype} | sample: {df2[col].iloc[0]}")
print()

# ============================================
# STEP 5: Align precipitation time to match
# ============================================
# ERA5 precipitation has 'time' + 'step'
# The actual valid time = time + step
# We need to create a matching time column

if "valid_time" in df2.columns:
    df2["merge_time"] = df2["valid_time"]
elif "step" in df2.columns:
    df2["merge_time"] = df2["time"] + df2["step"]
else:
    df2["merge_time"] = df2["time"]

# Round to daily
df2["merge_date"] = pd.to_datetime(df2["merge_time"]).dt.date
df1["merge_date"] = pd.to_datetime(df1["time"]).dt.date

# ============================================
# STEP 6: Merge
# ============================================
print("Merging datasets...")
df = pd.merge(
    df1,
    df2[["merge_date", "latitude", "longitude", "tp"]],
    on=["merge_date", "latitude", "longitude"],
    how="left"
)

# Clean up
df = df.drop(columns=["merge_date"], errors="ignore")

# ============================================
# STEP 7: Save
# ============================================
df.to_csv("era5_all_years.csv", index=False)

print()
print("=" * 50)
print("DONE!")
print("=" * 50)
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"TP NaN count: {df['tp'].isna().sum()} out of {len(df)}")
print(f"TP valid count: {df['tp'].notna().sum()}")
print()
print("Sample data:")
print(df.head(10))
print()
print("Statistics:")
print(df[["t2m", "u10", "v10", "sp", "swvl1", "tp"]].describe())