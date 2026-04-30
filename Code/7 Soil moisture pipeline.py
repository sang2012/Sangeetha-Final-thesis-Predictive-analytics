#=================================================================
#step7_soil_moisture
#=================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SOIL MOISTURE PREDICTION — COMPLETE PIPELINE")
print("="*60)

print("\nLoading data...")
df = pd.read_csv('era5_features.csv', parse_dates=['time'])
df = df.sample(n=1000000, random_state=42).sort_values('time').reset_index(drop=True)
print(f"Using: {len(df)} rows")

target = 'swvl1'
features = [
    'latitude', 'longitude', 'month', 'day_of_year',
    'u10', 'v10', 'sp_hpa', 'tp_mm', 't2m_celsius', 'wind_speed',
    'swvl1_lag1', 'swvl1_lag7', 'swvl1_lag30',
    'swvl1_roll7_mean', 'swvl1_roll30_mean',
    't2m_celsius_lag1', 'tp_mm_lag1', 'swvl1_anomaly'
]

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ============================================
# 1. TRAIN ALL 6 MODELS
# ============================================
print("\n" + "="*60)
print("PART 1: MODEL TRAINING (RMSE, MAE, R²)")
print("="*60)

models = [
    ('C1', 'Linear Regression', LinearRegression(), True),
    ('C2', 'SVR', SVR(kernel='rbf', C=10), True),
    ('C3', 'Decision Tree', DecisionTreeRegressor(max_depth=15, random_state=42), False),
    ('C4', 'Random Forest', RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1), False),
    ('C5', 'XGBoost', XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42), False),
    ('C6', 'KNN', KNeighborsRegressor(n_neighbors=5, n_jobs=-1), True),
]

results = []
best_model = None
best_r2 = -999

for exp, name, model, use_scaled in models:
    print(f"\n{exp}: {name}...")

    if name == 'SVR':
        idx = np.random.RandomState(42).choice(len(X_train_sc), 30000, replace=False)
        model.fit(X_train_sc[idx], y_train.iloc[idx])
        pred = model.predict(X_test_sc)
    elif use_scaled:
        model.fit(X_train_sc, y_train)
        pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    results.append({'Exp': exp, 'Model': name, 'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4), 'R2': round(r2, 4), 'MAPE': 'N/A'})
    print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name
        best_pred = pred

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
results_df.to_csv('results_soil_moisture.csv', index=False)
print(f"\n✅ Saved: results_soil_moisture.csv")
print(f"Best model: {best_name} (R² = {best_r2:.4f})")

# ============================================
# 2. SHAP ANALYSIS
# ============================================
print("\n" + "="*60)
print("PART 2: SHAP EXPLAINABILITY")
print("="*60)

xgb_model = XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

sample = X_test.sample(n=5000, random_state=42)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(sample)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
plt.title("Feature Importance — Soil Moisture (XGBoost)", fontsize=14)
plt.tight_layout()
plt.savefig('shap_bar_swvl1.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_bar_swvl1.png")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, sample, show=False)
plt.title("SHAP Summary — Soil Moisture (XGBoost)", fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_swvl1.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_summary_swvl1.png")

importance = pd.DataFrame({
    'Feature': features,
    'Mean_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)
importance.to_csv('shap_importance_swvl1.csv', index=False)
print("✅ Saved: shap_importance_swvl1.csv")

print("\nTop 5 features:")
for _, row in importance.head().iterrows():
    print(f"  {row['Feature']:30s} | SHAP: {row['Mean_SHAP']:.4f}")

# ============================================
# 3. PLOTS
# ============================================
print("\n" + "="*60)
print("PART 3: VISUALISATIONS")
print("="*60)

fig, ax = plt.subplots(figsize=(8, 8))
sample_idx = np.random.choice(len(y_test), 10000, replace=False)
ax.scatter(y_test.iloc[sample_idx], best_pred[sample_idx], alpha=0.3, s=5, color='green')
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax.set_xlabel('Actual Soil Moisture (m³/m³)', fontsize=12)
ax.set_ylabel('Predicted Soil Moisture (m³/m³)', fontsize=12)
ax.set_title(f'Predicted vs Actual — Soil Moisture ({best_name})', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_predicted_vs_actual_soil_moisture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_predicted_vs_actual_soil_moisture.png")

ts_df = pd.DataFrame({
    'date': df['time'].iloc[-len(y_test):].values,
    'actual': y_test.values,
    'predicted': best_pred
})
ts_df['date'] = pd.to_datetime(ts_df['date']).dt.date
daily = ts_df.groupby('date')[['actual', 'predicted']].mean().reset_index()
daily['date'] = pd.to_datetime(daily['date'])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily['date'], daily['actual'], color='green', linewidth=1, label='Actual', alpha=0.8)
ax.plot(daily['date'], daily['predicted'], color='red', linewidth=1, label='Predicted', alpha=0.8)
ax.set_title(f'Actual vs Predicted — Soil Moisture ({best_name})', fontsize=14)
ax.set_ylabel('m³/m³')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_timeseries_soil_moisture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_timeseries_soil_moisture.png")

# ============================================================
# CROSS-VALIDATION (OPTIMIZED)
# ============================================================
print("\nRunning 5-fold time-series cross-validation...")

tscv = TimeSeriesSplit(n_splits=5)
cv_results = []

# ✅ Use last 50k rows (time-consistent)
X_cv = X.tail(50000)
y_cv = y.tail(50000)

for exp, name, model, use_scaled in models:
    print(f"  CV: {name}")

    if use_scaled:
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
    else:
        estimator = model

    # ✅ Reduce heavy models only
    if name in ["SVR", "KNN"]:
        X_used, y_used = X_cv, y_cv
    else:
        X_used, y_used = X, y

    r2_scores = cross_val_score(estimator, X_used, y_used, cv=tscv, scoring="r2", n_jobs=-1)
    mae_scores = -cross_val_score(estimator, X_used, y_used, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)

    cv_results.append({
        "Exp": exp,
        "Model": name,
        "CV_R2_Mean": round(r2_scores.mean(), 4),
        "CV_R2_Std": round(r2_scores.std(), 4),
        "CV_MAE_Mean": round(mae_scores.mean(), 4),
        "CV_MAE_Std": round(mae_scores.std(), 4)
    })

cv_df = pd.DataFrame(cv_results).sort_values("CV_R2_Mean", ascending=False)
cv_df.to_csv("cv_results_soil_moisture.csv", index=False)

print("✅ Saved: cv_results_soil_moisture.csv")

# ============================================================
# SEASONAL EVALUATION (ALREADY OPTIMAL)
# ============================================================
print("\nRunning seasonal evaluation for best model...")

season_df = pd.DataFrame({
    "time": time_test,
    "actual": y_test.values,
    "predicted": best_pred
})

season_df["month"] = pd.to_datetime(season_df["time"]).dt.month
season_df["season"] = season_df["month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
})

seasonal_results = []

for season in ["Winter", "Spring", "Summer", "Autumn"]:
    subset = season_df[season_df["season"] == season]

    rmse = np.sqrt(mean_squared_error(subset["actual"], subset["predicted"]))
    mae = mean_absolute_error(subset["actual"], subset["predicted"])
    r2 = r2_score(subset["actual"], subset["predicted"])

    seasonal_results.append({
        "Season": season,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Count": len(subset)
    })

seasonal_df = pd.DataFrame(seasonal_results)
seasonal_df.to_csv("seasonal_results_soil_moisture.csv", index=False)

print("✅ Saved: seasonal_results_soil_moisture.csv")
print("\nSeasonal evaluation:")
print(seasonal_df)

print("\n" + "="*60)
print("SOIL MOISTURE PIPELINE COMPLETE!")
print("="*60)


# ============================================================
# TRAIN BEST MODEL (Random Forest)
# ============================================================
time_test = df["time"].iloc[-len(y_test):].reset_index(drop=True)

print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

# ============================================================
# METRICS
# ============================================================
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

# ============================================================
# RESIDUALS
# ============================================================
residuals = y_test.values - pred

sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Histogram
sns.histplot(residuals, bins=60, ax=axes[0],
             color="steelblue", edgecolor="black")
axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
axes[0].set_title("Residual Distribution — Soil Moisture")
axes[0].set_xlabel("Residual")
axes[0].set_ylabel("Frequency")

# Residuals over time
axes[1].scatter(time_test, residuals, s=8, alpha=0.5, color="steelblue")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title("Residuals Over Time — Soil Moisture")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Residual")

plt.tight_layout()
plt.savefig("residuals_soil_moisture.png", dpi=300)
plt.show()

print("✅ Saved: residuals_soil_moisture.png")