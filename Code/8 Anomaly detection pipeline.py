#===================================
# Step8_anomaly_all
#===================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, precision_recall_curve, roc_curve, auc)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ANOMALY DETECTION — COMPLETE PIPELINE")
print("="*60)

print("\nLoading data...")
df = pd.read_csv('era5_features.csv', parse_dates=['time'])
df = df.sample(n=1000000, random_state=42).sort_values('time').reset_index(drop=True)
print(f"Using: {len(df)} rows")

features = [
    'latitude', 'longitude', 'month', 'day_of_year',
    'u10', 'v10', 'sp_hpa', 'swvl1', 'tp_mm', 'wind_speed',
    't2m_celsius_lag1', 't2m_celsius_lag7', 't2m_celsius_lag30',
    't2m_celsius_roll7_mean', 't2m_celsius_roll30_mean',
    'tp_mm_lag1', 'swvl1_lag1'
]

def run_full_anomaly(df, anomaly_col, label_name, extreme_type):
    print(f"\n{'='*60}")
    print(f"ANOMALY DETECTION: {label_name}")
    print(f"{'='*60}")

    anom_std = df[anomaly_col].std()
    if extreme_type == 'both':
        df[f'{label_name}_extreme'] = (df[anomaly_col].abs() > 2 * anom_std).astype(int)
    else:
        df[f'{label_name}_extreme'] = (df[anomaly_col] > 2 * anom_std).astype(int)

    X = df[features]
    y = df[f'{label_name}_extreme']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"\nNormal: {(y_train==0).sum()} | Anomaly: {(y_train==1).sum()} ({y_train.mean()*100:.1f}%)")

    # ---- D1: Statistical Baseline ----
    print(f"\n--- D1: Statistical Baseline (±2σ) ---")
    test_anomaly = df.iloc[-len(y_test):][anomaly_col]
    test_std = test_anomaly.std()
    if extreme_type == 'both':
        e1_pred = (test_anomaly.abs() > 2 * test_std).astype(int)
    else:
        e1_pred = (test_anomaly > 2 * test_std).astype(int)

    e1_scores = test_anomaly.abs() / (2 * test_std)
    e1_scores = e1_scores.clip(0, 1)

    p = precision_score(y_test, e1_pred, zero_division=0)
    r = recall_score(y_test, e1_pred, zero_division=0)
    f1 = f1_score(y_test, e1_pred, zero_division=0)
    ba = balanced_accuracy_score(y_test, e1_pred)
    roc = roc_auc_score(y_test, e1_scores)
    print(f"  Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f} | Balanced Acc: {ba:.2f} | ROC-AUC: {roc:.4f}")

    # ---- D3: One-Class SVM ----
    print(f"\n--- D3: One-Class SVM ---")
    X_train_normal = X_train[y_train == 0]
    sample_size = min(30000, len(X_train_normal))
    sample_idx = np.random.RandomState(42).choice(len(X_train_normal), sample_size, replace=False)
    X_sample_sc = scaler.fit_transform(X_train_normal.iloc[sample_idx])
    X_test_sc2 = scaler.transform(X_test)

    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    ocsvm.fit(X_sample_sc)
    raw_pred = ocsvm.predict(X_test_sc2)
    e3_pred = (raw_pred == -1).astype(int)
    e3_scores = -ocsvm.decision_function(X_test_sc2)

    p = precision_score(y_test, e3_pred, zero_division=0)
    r = recall_score(y_test, e3_pred, zero_division=0)
    f1 = f1_score(y_test, e3_pred, zero_division=0)
    ba = balanced_accuracy_score(y_test, e3_pred)
    roc = roc_auc_score(y_test, e3_scores)
    print(f"  Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f} | Balanced Acc: {ba:.2f} | ROC-AUC: {roc:.4f}")

    # ---- BEFORE SMOTE: D2, D4-D7 ----
    print(f"\n--- BEFORE SMOTE ---")
    clf_models = [
        ('D2', 'Logistic Regression', LogisticRegression(max_iter=1000, random_state=42), True),
        ('D4', 'Decision Tree', DecisionTreeClassifier(max_depth=15, random_state=42), False),
        ('D5', 'Random Forest', RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1), False),
        ('D6', 'XGBoost', XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, eval_metric='logloss'), False),
        ('D7', 'KNN', KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
    ]

    for exp, name, model, use_scaled in clf_models:
        if use_scaled:
            model.fit(X_train_sc, y_train)
            pred = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        p = precision_score(y_test, pred, zero_division=0)
        r = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        print(f"  {exp} {name:25s} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}")

    # ---- AFTER SMOTE: D2, D4-D7 ----
    print(f"\n--- AFTER SMOTE ---")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    X_train_sm_sc = scaler.fit_transform(X_train_sm)
    X_test_sc3 = scaler.transform(X_test)
    print(f"  Normal: {(y_train_sm==0).sum()} | Anomaly: {(y_train_sm==1).sum()}")

    clf_models_smote = [
        ('D2', 'Logistic Regression', LogisticRegression(max_iter=1000, random_state=42), True),
        ('D4', 'Decision Tree', DecisionTreeClassifier(max_depth=15, random_state=42), False),
        ('D5', 'Random Forest', RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1), False),
        ('D6', 'XGBoost', XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, eval_metric='logloss'), False),
        ('D7', 'KNN', KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
    ]

    for exp, name, model, use_scaled in clf_models_smote:
        if use_scaled:
            model.fit(X_train_sm_sc, y_train_sm)
            pred = model.predict(X_test_sc3)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test_sc3)[:, 1]
        else:
            model.fit(X_train_sm, y_train_sm)
            pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]

        p = precision_score(y_test, pred, zero_division=0)
        r = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        ba = balanced_accuracy_score(y_test, pred)

        roc_str = ""
        if exp in ['D5', 'D6']:
            roc = roc_auc_score(y_test, proba)
            roc_str = f" | ROC-AUC: {roc:.4f}"

        print(f"  {exp} {name:25s} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f} | Balanced Acc: {ba:.2f}{roc_str}")

        if exp == 'D6':
            cm = confusion_matrix(y_test, pred)
            print(f"\n  Confusion Matrix ({name}):")
            print(f"    True Positive:  {cm[1][1]:,}")
            print(f"    False Negative: {cm[1][0]:,}")
            print(f"    True Negative:  {cm[0][0]:,}")
            print(f"    False Positive: {cm[0][1]:,}")

# Run all 3 variables
run_full_anomaly(df, 't2m_anomaly', 'Temperature', 'both')
run_full_anomaly(df, 'tp_anomaly', 'Precipitation', 'high')
run_full_anomaly(df, 'swvl1_anomaly', 'Soil Moisture', 'both')

# ============================================
# THRESHOLD TUNING (XGBoost, Temperature)
# ============================================
print(f"\n{'='*60}")
print("THRESHOLD TUNING (XGBoost, Temperature)")
print(f"{'='*60}")

t2m_std = df['t2m_anomaly'].std()
df['t2m_extreme'] = (df['t2m_anomaly'].abs() > 2 * t2m_std).astype(int)
X = df[features]
y = df['t2m_extreme']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
xgb = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb.fit(X_sm, y_sm)
proba = xgb.predict_proba(X_te)[:, 1]

print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 48)
for t in [0.50, 0.40, 0.35, 0.30, 0.25]:
    pred = (proba >= t).astype(int)
    p = precision_score(y_te, pred, zero_division=0)
    r = recall_score(y_te, pred, zero_division=0)
    f1 = f1_score(y_te, pred, zero_division=0)
    print(f"{t:<12.2f} {p:<12.2f} {r:<12.2f} {f1:<12.2f}")

print(f"\n{'='*60}")
print("ANOMALY DETECTION PIPELINE COMPLETE!")
print(f"{'='*60}")
