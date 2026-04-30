"""
=================================================================
STEP 5: TEMPERATURE PREDICTION — COMPLETE PIPELINE
=================================================================
Script: step5_temperature.py
Target: 2-metre temperature (t2m_celsius)
Models: Linear Regression, SVR, Decision Tree, Random Forest,
        XGBoost, KNN
Metrics: RMSE, MAE, R², MAPE
Includes: Cross-validation, Seasonal evaluation, SHAP analysis,
          Visualisations
Input: era5_features.csv
Output: results_temperature.csv
        shap_bar_t2m_celsius.png
        shap_summary_t2m_celsius.png
        shap_importance_t2m_celsius.csv
        viz_predicted_vs_actual_temperature.png
        viz_residuals_temperature.png
        viz_feature_importance_temperature.png
        viz_model_comparison_temperature.png
=================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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


# =================================================================
# CONFIGURATION
# =================================================================
INPUT_FILE = 'era5_features.csv'
SAMPLE_SIZE = 1000000
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = 't2m_celsius'

FEATURES = [
    'latitude', 'longitude', 'month', 'day_of_year',
    'u10', 'v10', 'sp_hpa', 'swvl1', 'tp_mm', 'wind_speed',
    't2m_celsius_lag1', 't2m_celsius_lag7', 't2m_celsius_lag30',
    't2m_celsius_roll7_mean', 't2m_celsius_roll30_mean',
    'tp_mm_lag1', 'swvl1_lag1', 't2m_anomaly'
]


# =================================================================
# HELPER FUNCTIONS
# =================================================================
def calc_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100


def print_header(title):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


# =================================================================
# 1. LOAD AND PREPARE DATA
# =================================================================
print_header("TEMPERATURE PREDICTION — COMPLETE PIPELINE")

print("\nLoading data...")
df = pd.read_csv(INPUT_FILE, parse_dates=['time'])
df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).sort_values('time').reset_index(drop=True)
print(f"Full dataset: {len(pd.read_csv(INPUT_FILE, nrows=1).columns)} columns")
print(f"Using sample: {len(df)} rows")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"Target: {TARGET}")
print(f"Features: {len(FEATURES)}")

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# =================================================================
# 2. TRAIN ALL 6 MODELS (RMSE, MAE, R², MAPE)
# =================================================================
print_header("PART 1: MODEL TRAINING")

models = [
    ('A1', 'Linear Regression',
     LinearRegression(), True),

    ('A2', 'SVR',
     SVR(kernel='rbf', C=10), True),

    ('A3', 'Decision Tree',
     DecisionTreeRegressor(max_depth=15, random_state=RANDOM_STATE), False),

    ('A4', 'Random Forest',
     RandomForestRegressor(n_estimators=50, max_depth=12,
                           random_state=RANDOM_STATE, n_jobs=-1), False),

    ('A5', 'XGBoost',
     XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1,
                  random_state=RANDOM_STATE), False),

    ('A6', 'KNN',
     KNeighborsRegressor(n_neighbors=5, n_jobs=-1), True),
]

results = []
best_model = None
best_r2 = -999
best_pred = None
best_name = None
trained_models = {}

for exp, name, model, use_scaled in models:
    print(f"\n{exp}: {name}...")

    # SVR needs sampling due to slow training
    if name == 'SVR':
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train_sc), 30000, replace=False
        )
        model.fit(X_train_sc[idx], y_train.iloc[idx])
        pred = model.predict(X_test_sc)
    elif use_scaled:
        model.fit(X_train_sc, y_train)
        pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    mape = calc_mape(y_test.values, pred)

    results.append({
        'Exp': exp,
        'Model': name,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2)
    })

    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    # Track best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name
        best_pred = pred

    # Store trained models for later use
    trained_models[name] = model

# Save results
results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
results_df.to_csv('results_temperature.csv', index=False)

print(f"\n✅ Saved: results_temperature.csv")
print(f"\nModel Rankings:")
print(results_df.to_string(index=False))
print(f"\n🏆 Best Model: {best_name} (R² = {best_r2:.4f})")


# =================================================================
# 3. CROSS-VALIDATION (5-Fold Time-Series)
# =================================================================
print_header("PART 2: CROSS-VALIDATION (5-Fold Time-Series)")

tscv = TimeSeriesSplit(n_splits=5)

cv_models = [
    ('Linear Regression', LinearRegression(), True),
    ('Decision Tree',
     DecisionTreeRegressor(max_depth=15, random_state=RANDOM_STATE), False),
    ('Random Forest',
     RandomForestRegressor(n_estimators=50, max_depth=12,
                           random_state=RANDOM_STATE, n_jobs=-1), False),
    ('KNN', KNeighborsRegressor(n_neighbors=5, n_jobs=-1), True),
    ('XGBoost',
     XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1,
                  random_state=RANDOM_STATE), False),
]

cv_results = []

for name, model, use_scaled in cv_models:
    fold_scores = {'RMSE': [], 'MAE': [], 'MAPE': [], 'R2': []}

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X.values), 1):
        X_tr, X_te = X.values[tr_idx], X.values[te_idx]
        y_tr, y_te = y.values[tr_idx], y.values[te_idx]

        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)

        if use_scaled:
            model.fit(X_tr_sc, y_tr)
            p = model.predict(X_te_sc)
        else:
            model.fit(X_tr, y_tr)
            p = model.predict(X_te)

        fold_scores['RMSE'].append(np.sqrt(mean_squared_error(y_te, p)))
        fold_scores['MAE'].append(mean_absolute_error(y_te, p))
        fold_scores['MAPE'].append(calc_mape(y_te, p))
        fold_scores['R2'].append(r2_score(y_te, p))

    cv_results.append({
        'Model': name,
        'RMSE_mean': round(np.mean(fold_scores['RMSE']), 4),
        'RMSE_std': round(np.std(fold_scores['RMSE']), 4),
        'MAE_mean': round(np.mean(fold_scores['MAE']), 4),
        'MAPE_mean': round(np.mean(fold_scores['MAPE']), 2),
        'R2_mean': round(np.mean(fold_scores['R2']), 4),
        'R2_std': round(np.std(fold_scores['R2']), 4),
    })

    print(f"  {name:25s} | R²: {np.mean(fold_scores['R2']):.4f} ± {np.std(fold_scores['R2']):.4f} | MAPE: {np.mean(fold_scores['MAPE']):.2f}%")

cv_df = pd.DataFrame(cv_results).sort_values('R2_mean', ascending=False)
cv_df.to_csv('cv_results_temperature.csv', index=False)
print(f"\n✅ Saved: cv_results_temperature.csv")


# =================================================================
# 4. SEASONAL EVALUATION
# =================================================================
print_header("PART 3: SEASONAL EVALUATION (Best Model: {})".format(best_name))

test_df = df.iloc[-len(y_test):].copy()
test_df['predicted'] = best_pred

print(f"\n{'Season':<12} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Count':>8}")
print("-" * 48)

for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    sdf = test_df[test_df['season'] == season]
    rmse = np.sqrt(mean_squared_error(sdf['t2m_celsius'], sdf['predicted']))
    mae = mean_absolute_error(sdf['t2m_celsius'], sdf['predicted'])
    r2 = r2_score(sdf['t2m_celsius'], sdf['predicted'])
    print(f"  {season:<10} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f} {len(sdf):>8}")


# =================================================================
# 5. SHAP ANALYSIS
# =================================================================
print_header("PART 4: SHAP EXPLAINABILITY (XGBoost)")

# Train fresh XGBoost for SHAP
xgb_model = XGBRegressor(
    n_estimators=100, max_depth=10, learning_rate=0.1,
    random_state=RANDOM_STATE
)
xgb_model.fit(X_train, y_train)

# Calculate SHAP values
print("Calculating SHAP values (5000 samples)...")
sample = X_test.sample(n=5000, random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(sample)

# SHAP Bar Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
plt.title("Feature Importance — Temperature (XGBoost)", fontsize=14)
plt.tight_layout()
plt.savefig('shap_bar_t2m_celsius.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_bar_t2m_celsius.png")

# SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, sample, show=False)
plt.title("SHAP Summary — Temperature (XGBoost)", fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_t2m_celsius.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_summary_t2m_celsius.png")

# Save SHAP importance
importance = pd.DataFrame({
    'Feature': FEATURES,
    'Mean_SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean_SHAP', ascending=False)
importance.to_csv('shap_importance_t2m_celsius.csv', index=False)
print("✅ Saved: shap_importance_t2m_celsius.csv")

print("\nTop 5 Features:")
for _, row in importance.head().iterrows():
    print(f"  {row['Feature']:30s} | SHAP: {row['Mean_SHAP']:.4f}")


# =================================================================
# 6. VISUALISATIONS
# =================================================================
print_header("PART 5: VISUALISATIONS")

# --- 6a. Predicted vs Actual Scatter ---
print("\nCreating predicted vs actual scatter...")
fig, ax = plt.subplots(figsize=(8, 8))
sample_idx = np.random.choice(len(y_test), 10000, replace=False)
ax.scatter(
    y_test.iloc[sample_idx], best_pred[sample_idx],
    alpha=0.3, s=5, color='blue'
)
ax.plot([-15, 40], [-15, 40], 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Temperature (°C)', fontsize=12)
ax.set_ylabel('Predicted Temperature (°C)', fontsize=12)
ax.set_title(f'Predicted vs Actual — Temperature ({best_name})', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_predicted_vs_actual_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_predicted_vs_actual_temperature.png")

# --- 6b. Residual Distribution ---
print("Creating residual distribution...")
residuals = y_test.values - best_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(residuals, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='red', linewidth=2, linestyle='--')
axes[0].set_title(f'Residual Distribution — Temperature ({best_name})', fontsize=12)
axes[0].set_xlabel('Residual (°C)')
axes[0].set_ylabel('Frequency')

test_dates = df['time'].iloc[-len(y_test):].values
axes[1].scatter(test_dates[::50], residuals[::50], s=3, alpha=0.5, color='steelblue')
axes[1].axhline(y=0, color='red', linewidth=1, linestyle='--')
axes[1].set_title(f'Residuals Over Time — Temperature ({best_name})', fontsize=12)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Residual (°C)')

plt.tight_layout()
plt.savefig('viz_residuals_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_residuals_temperature.png")

# --- 6c. Feature Importance Comparison ---
print("Creating feature importance comparison...")
dt_model = DecisionTreeRegressor(max_depth=15, random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(
    n_estimators=50, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
)
rf_model.fit(X_train, y_train)

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
model_imps = [
    ('Decision Tree', dt_model.feature_importances_, '#e74c3c'),
    ('Random Forest', rf_model.feature_importances_, '#2ecc71'),
    ('XGBoost', xgb_model.feature_importances_, '#3498db')
]

for ax, (name, imp, color) in zip(axes, model_imps):
    sorted_idx = np.argsort(imp)
    ax.barh(range(len(FEATURES)), imp[sorted_idx], color=color, alpha=0.8)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([FEATURES[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title(name, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

fig.suptitle(
    'Native Feature Importance — Temperature (All Ensemble Models)',
    fontsize=16, y=1.02
)
plt.tight_layout()
plt.savefig('viz_feature_importance_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_feature_importance_temperature.png")

# --- 6d. Model Comparison Bar Chart ---
print("Creating model comparison chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rdf = results_df.sort_values('R2', ascending=True)
colors = ['#e74c3c', '#e67e22', '#3498db', '#2980b9', '#27ae60', '#2ecc71']

axes[0].barh(rdf['Model'], rdf['R2'], color=colors)
axes[0].set_xlim(0.93, 1.0)
axes[0].set_xlabel('R² Score')
axes[0].set_title('Model Comparison — R² (Temperature)')
for i, v in enumerate(rdf['R2']):
    axes[0].text(v + 0.001, i, f'{v:.4f}', va='center')

axes[1].barh(rdf['Model'], rdf['RMSE'], color=colors)
axes[1].set_xlabel('RMSE (°C)')
axes[1].set_title('Model Comparison — RMSE (Temperature)')
for i, v in enumerate(rdf['RMSE']):
    axes[1].text(v + 0.01, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.savefig('viz_model_comparison_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_model_comparison_temperature.png")

# --- 6e. Time-Series Overlay ---
print("Creating time-series overlay...")
ts_df = pd.DataFrame({
    'date': df['time'].iloc[-len(y_test):].values,
    'actual': y_test.values,
    'predicted': best_pred
})
ts_df['date'] = pd.to_datetime(ts_df['date']).dt.date
daily = ts_df.groupby('date')[['actual', 'predicted']].mean().reset_index()
daily['date'] = pd.to_datetime(daily['date'])

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily['date'], daily['actual'], color='blue', linewidth=1,
        label='Actual', alpha=0.8)
ax.plot(daily['date'], daily['predicted'], color='red', linewidth=1,
        label='Predicted', alpha=0.8)
ax.set_title(
    f'Actual vs Predicted Temperature — Daily Average ({best_name})',
    fontsize=14
)
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_timeseries_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_timeseries_temperature.png")


# =================================================================
# SUMMARY
# =================================================================
print_header("TEMPERATURE PIPELINE — COMPLETE!")

print("\nFiles generated:")
print("  📊 results_temperature.csv")
print("  📊 cv_results_temperature.csv")
print("  📊 shap_importance_t2m_celsius.csv")
print("  📈 shap_bar_t2m_celsius.png")
print("  📈 shap_summary_t2m_celsius.png")
print("  📈 viz_predicted_vs_actual_temperature.png")
print("  📈 viz_residuals_temperature.png")
print("  📈 viz_feature_importance_temperature.png")
print("  📈 viz_model_comparison_temperature.png")
print("  📈 viz_timeseries_temperature.png")

print(f"\n🏆 Best Model: {best_name}")
print(f"   RMSE: {results_df.iloc[0]['RMSE']}")
print(f"   MAE:  {results_df.iloc[0]['MAE']}")
print(f"   R²:   {results_df.iloc[0]['R2']}")
print(f"   MAPE: {results_df.iloc[0]['MAPE']}%")