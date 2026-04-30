import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# ============================================================
# SETTINGS
# ============================================================
INPUT_FILE = "era5_features.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 1000000

# Feature sets (adjust if needed to match your original anomaly code)
TEMP_FEATURES = [
    'latitude', 'longitude', 'month', 'day_of_year',
    'u10', 'v10', 'sp_hpa', 'swvl1', 'tp_mm', 'wind_speed',
    't2m_celsius_lag1', 't2m_celsius_lag7', 't2m_celsius_lag30',
    't2m_celsius_roll7_mean', 't2m_celsius_roll30_mean',
    'tp_mm_lag1', 'swvl1_lag1', 't2m_anomaly'
]

RAIN_FEATURES = [
    'latitude', 'longitude', 'month', 'day_of_year',
    'u10', 'v10', 'sp_hpa', 'swvl1', 't2m_celsius', 'wind_speed',
    'tp_mm_lag1', 'tp_mm_lag7', 'tp_mm_lag30',
    'tp_mm_roll7_mean', 'tp_mm_roll30_mean',
    't2m_celsius_lag1', 'swvl1_lag1', 'tp_anomaly'
]

SOIL_FEATURES = [
    'latitude', 'longitude', 'month', 'day_of_year',
    'u10', 'v10', 'sp_hpa', 'tp_mm', 't2m_celsius', 'wind_speed',
    'swvl1_lag1', 'swvl1_lag7', 'swvl1_lag30',
    'swvl1_roll7_mean', 'swvl1_roll30_mean',
    'tp_mm_lag1', 't2m_celsius_lag1', 'swvl1_anomaly'
]

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).sort_values("time").reset_index(drop=True)
print(f"Using: {len(df)} rows")

# ============================================================
# HELPER FUNCTION
# ============================================================
def create_anomaly_label(series):
    mean_val = series.mean()
    std_val = series.std()
    lower = mean_val - 2 * std_val
    upper = mean_val + 2 * std_val
    return ((series < lower) | (series > upper)).astype(int)

def run_before_smote_confmat(target_col, features, variable_name, save_name):
    print(f"\nRunning before-SMOTE confusion matrix for {variable_name}...")

     # create labels
    y = create_anomaly_label(df[target_col])
    X = df[features]

    # chronological split
    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=TEST_SIZE, shuffle=False
    )

    # train only XGBoost before SMOTE
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"{variable_name} confusion matrix:")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # plot
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix Before SMOTE — {variable_name} (XGBoost)")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# RUN FOR ALL THREE VARIABLES
# ============================================================
run_before_smote_confmat(
    target_col="t2m_celsius",
    features=TEMP_FEATURES,
    variable_name="Temperature",
    save_name="viz_confmat_before_smote_temperature.png"
)

run_before_smote_confmat(
    target_col="tp_mm",
    features=RAIN_FEATURES,
    variable_name="Precipitation",
    save_name="viz_confmat_before_smote_precipitation.png"
)

run_before_smote_confmat(
    target_col="swvl1",
    features=SOIL_FEATURES,
    variable_name="Soil Moisture",
    save_name="viz_confmat_before_smote_soil_moisture.png"
)

print("\nFiles saved:")
print(" - viz_confmat_before_smote_temperature.png")
print(" - viz_confmat_before_smote_precipitation.png")
print(" - viz_confmat_before_smote_soil_moisture.png")