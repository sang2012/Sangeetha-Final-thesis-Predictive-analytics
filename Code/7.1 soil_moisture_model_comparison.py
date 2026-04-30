import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CREATE DATAFRAME (your results)
# ============================================================
data = [
    ["C5", "XGBoost", 0.0061, 0.0024, 0.9987],
    ["C4", "Random Forest", 0.0075, 0.0029, 0.9981],
    ["C3", "Decision Tree", 0.0088, 0.0032, 0.9973],
    ["C1", "Linear Regression", 0.0092, 0.0046, 0.9971],
    ["C6", "KNN", 0.0121, 0.0055, 0.9950],
    ["C2", "SVR", 0.0469, 0.0403, 0.9248],
]

df = pd.DataFrame(data, columns=["Exp", "Model", "RMSE", "MAE", "R2"])

# ============================================================
# PLOT
# ============================================================
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Sort data
rmse_df = df.sort_values("RMSE", ascending=True)
mae_df = df.sort_values("MAE", ascending=True)
r2_df = df.sort_values("R2", ascending=True)

# 🔹 RMSE (Blues)
sns.barplot(
    data=rmse_df, y="Model", x="RMSE",
    ax=axes[0], palette="Blues_r"
)
axes[0].set_title("Model Comparison — RMSE")
axes[0].set_xlabel("RMSE")

for i, v in enumerate(rmse_df["RMSE"]):
    axes[0].text(v, i, f" {v:.4f}", va="center")

# 🔹 MAE (Greens)
sns.barplot(
    data=mae_df, y="Model", x="MAE",
    ax=axes[1], palette="Greens_r"
)
axes[1].set_title("Model Comparison — MAE")
axes[1].set_xlabel("MAE")

for i, v in enumerate(mae_df["MAE"]):
    axes[1].text(v, i, f" {v:.4f}", va="center")

# 🔹 R2 (Oranges)
sns.barplot(
    data=r2_df, y="Model", x="R2",
    ax=axes[2], palette="Oranges"
)
axes[2].set_title("Model Comparison — R²")
axes[2].set_xlabel("R² Score")

for i, v in enumerate(r2_df["R2"]):
    axes[2].text(v, i, f" {v:.4f}", va="center")

plt.tight_layout()
plt.savefig("model_comparison_colored.png", dpi=300)
plt.show()
print("✅ Saved: model_comparison.png")