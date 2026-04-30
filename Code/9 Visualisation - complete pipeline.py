# ============================================
# step9_visualisations
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ALL VISUALISATIONS — COMPLETE PIPELINE")
print("="*60)

print("\nLoading data...")
df = pd.read_csv('era5_features.csv', parse_dates=['time'])
df = df.sample(n=1000000, random_state=42).sort_values('time').reset_index(drop=True)
print(f"Using: {len(df)} rows")

# ============================================
# 1. TIME-SERIES
# ============================================
print("\n1. Time-series plot...")
monthly = df.groupby(df['time'].dt.to_period('M')).agg({
    't2m_celsius': 'mean', 'tp_mm': 'mean', 'swvl1': 'mean'
}).reset_index()
monthly['time'] = monthly['time'].dt.to_timestamp()

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(monthly['time'], monthly['t2m_celsius'], color='red', linewidth=1.5)
axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Monthly Average Temperature (2010–2025)')
axes[0].grid(True, alpha=0.3)
axes[1].bar(monthly['time'], monthly['tp_mm'], color='blue', width=25, alpha=0.7)
axes[1].set_ylabel('Precipitation (mm)')
axes[1].set_title('Monthly Average Precipitation (2010–2025)')
axes[1].grid(True, alpha=0.3)
axes[2].plot(monthly['time'], monthly['swvl1'], color='green', linewidth=1.5)
axes[2].set_ylabel('Soil Moisture (m³/m³)')
axes[2].set_title('Monthly Average Soil Moisture (2010–2025)')
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_timeseries.png")

# ============================================
# 2. SEASONAL BOXPLOTS
# ============================================
print("2. Seasonal boxplots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
df.boxplot(column='t2m_celsius', by='month', ax=axes[0], grid=False)
axes[0].set_title('Temperature by Month')
axes[0].set_xticklabels(month_labels, rotation=45)
fig.suptitle('Boxplot grouped by month')
df.boxplot(column='tp_mm', by='month', ax=axes[1], grid=False)
axes[1].set_title('Precipitation by Month')
axes[1].set_xticklabels(month_labels, rotation=45)
df.boxplot(column='swvl1', by='month', ax=axes[2], grid=False)
axes[2].set_title('Soil Moisture by Month')
axes[2].set_xticklabels(month_labels, rotation=45)
plt.tight_layout()
plt.savefig('viz_seasonal_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_seasonal_boxplots.png")

# ============================================
# 3. SPATIAL HEATMAP
# ============================================
print("3. Spatial heatmap...")
spatial = df.groupby(['latitude', 'longitude'])['t2m_celsius'].mean().reset_index()
pivot = spatial.pivot(index='latitude', columns='longitude', values='t2m_celsius')
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values, cmap='RdYlBu_r', shading='auto')
plt.colorbar(im, ax=ax, label='Temperature (°C)')
ax.set_title('Mean Temperature Across UK (2010–2025)', fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig('viz_spatial_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_spatial_temperature.png")

# ============================================
# 4. CORRELATION HEATMAP
# ============================================
print("4. Correlation heatmap...")
corr_cols = ['t2m_celsius', 'tp_mm', 'sp_hpa', 'u10', 'v10', 'swvl1', 'wind_speed']
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha='right')
ax.set_yticklabels(corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=9)
ax.set_title('Correlation Heatmap — Climate Variables', fontsize=14)
plt.tight_layout()
plt.savefig('viz_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_correlation_heatmap.png")

# ============================================
# 5. ANOMALY TIMELINE
# ============================================
print("5. Anomaly timeline...")
daily_avg = df.groupby(df['time'].dt.date)['t2m_anomaly'].mean().reset_index()
daily_avg['time'] = pd.to_datetime(daily_avg['time'])
colors = ['red' if v > 0 else 'blue' for v in daily_avg['t2m_anomaly']]
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(daily_avg['time'], daily_avg['t2m_anomaly'], color=colors, width=1, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('Daily Temperature Anomaly (2010–2025)', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Anomaly (°C)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('viz_anomaly_timeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_anomaly_timeline.png")

# ============================================
# 6. DISTRIBUTIONS
# ============================================
print("6. Distribution plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(df['t2m_celsius'], bins=50, color='red', alpha=0.7, edgecolor='black')
axes[0].set_title('Temperature Distribution')
axes[0].set_xlabel('°C')
axes[1].hist(df['tp_mm'], bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[1].set_title('Precipitation Distribution')
axes[1].set_xlabel('mm')
axes[2].hist(df['swvl1'], bins=50, color='green', alpha=0.7, edgecolor='black')
axes[2].set_title('Soil Moisture Distribution')
axes[2].set_xlabel('m³/m³')
plt.tight_layout()
plt.savefig('viz_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_distributions.png")

# ============================================
# 7. WIND ROSE
# ============================================
print("7. Wind rose...")
df['wind_dir'] = (np.degrees(np.arctan2(-df['u10'], -df['v10'])) + 360) % 360
dir_bins = np.arange(0, 381, 22.5)
speed_bins = [0, 2, 4, 6, 8, 10, 25]
speed_labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '>10']
speed_colors = ['#2196F3', '#4CAF50', '#FFEB3B', '#FF9800', '#F44336', '#9C27B0']
df['dir_bin'] = pd.cut(df['wind_dir'], bins=dir_bins, labels=range(16), include_lowest=True)
df['speed_bin'] = pd.cut(df['wind_speed'], bins=speed_bins, labels=speed_labels, include_lowest=True)
dir_labels = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
width = 2*np.pi/16
total = len(df)

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
bottom = np.zeros(16)
for sl, color in zip(speed_labels, speed_colors):
    values = np.array([len(df[(df['dir_bin']==d)&(df['speed_bin']==sl)])/total*100 for d in range(16)])
    ax.bar(angles, values, width=width, bottom=bottom, color=color, edgecolor='white', linewidth=0.5, label=sl, alpha=0.85)
    bottom += values
ax.set_xticks(angles)
ax.set_xticklabels(dir_labels)
ax.set_title('Wind Rose — UK (2010–2025)', fontsize=14, pad=20)
ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), title='m/s')
plt.tight_layout()
plt.savefig('viz_wind_rose.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_wind_rose.png")

# ============================================
# 8. DASHBOARDS
# ============================================
print("8. Composite dashboards...")
for var, var_name, color, cmap in [
    ('t2m_celsius', 'Temperature', 'red', 'RdYlBu_r'),
    ('tp_mm', 'Precipitation', 'blue', 'Blues'),
    ('swvl1', 'Soil Moisture', 'green', 'YlGn')]:

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Climate Dashboard — {var_name} (2010–2025)', fontsize=16, y=1.02)

    m = df.groupby(df['time'].dt.to_period('M'))[var].mean().reset_index()
    m['time'] = m['time'].dt.to_timestamp()
    axes[0,0].plot(m['time'], m[var], color=color)
    axes[0,0].set_title(f'Monthly Mean {var_name}')
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].hist(df[var], bins=50, color=color, alpha=0.7, edgecolor='black')
    axes[0,1].set_title(f'{var_name} Distribution')

    season_order = ['Winter','Spring','Summer','Autumn']
    data = [df[df['season']==s][var].values for s in season_order]
    axes[0,2].boxplot(data, labels=season_order, patch_artist=True)
    axes[0,2].set_title(f'{var_name} by Season')

    anom_col = f'{var.split("_")[0]}_anomaly' if var != 'swvl1' else 'swvl1_anomaly'
    if var == 't2m_celsius': anom_col = 't2m_anomaly'
    if var == 'tp_mm': anom_col = 'tp_anomaly'
    d = df.groupby(df['time'].dt.date)[anom_col].mean().reset_index()
    d['time'] = pd.to_datetime(d['time'])
    c = [color if v > 0 else 'grey' for v in d[anom_col]]
    axes[1,0].bar(d['time'], d[anom_col], color=c, width=1, alpha=0.7)
    axes[1,0].set_title(f'{var_name} Anomaly')

    a = df.groupby('year')[var].mean().reset_index()
    axes[1,1].bar(a['year'], a[var], color=color, edgecolor='black')
    axes[1,1].set_title(f'Annual Mean {var_name}')

    sp = df.groupby(['latitude','longitude'])[var].mean().reset_index()
    pv = sp.pivot(index='latitude', columns='longitude', values=var)
    im = axes[1,2].pcolormesh(pv.columns, pv.index, pv.values, cmap=cmap, shading='auto')
    plt.colorbar(im, ax=axes[1,2])
    axes[1,2].set_title(f'Spatial Mean {var_name}')

    plt.tight_layout()
    fname = f'viz_dashboard_{var_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fname}")

# ============================================
# 9. FRAMEWORK DIAGRAMS
# ============================================
print("\n9. Framework diagrams...")

fig, ax = plt.subplots(figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 24)
ax.axis('off')
fig.patch.set_facecolor('white')

grey='#7f8c8d'; blue='#3498db'; green='#2ecc71'; orange='#f39c12'; red='#e74c3c'; purple='#9b59b6'; dark='#2c3e50'

def draw_box(ax, x, y, w, h, text, color, fs=13):
    box = mpatches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.2", facecolor=color, edgecolor='white', linewidth=2.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, fontweight='bold', color='white')

def draw_arrow(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=3))

ax.text(7, 23.3, 'Research Workflow', ha='center', fontsize=22, fontweight='bold', color=dark)
ax.text(7, 22.8, 'Climate Prediction & Anomaly Detection', ha='center', fontsize=16, color='#555')

bx=3; bw=8; bh=1.0; gap=0.5; cx=bx+bw/2
steps = [
    ('Literature Review\n& Problem Identification', grey, 'Ch 1–2'),
    ('ERA5 Data Selection\n(Copernicus CDS)', blue, 'Ch 3'),
    ('Data Preprocessing\n& Cleaning', blue, 'Ch 3'),
    ('Exploratory Data Analysis\n& Visualisation', blue, 'Ch 4'),
    ('Feature Engineering\n(Lags, Rolling, Anomalies)', green, 'Ch 3–4'),
    ('Class Balancing\n(SMOTE)', green, 'Ch 3–4'),
    ('Model Training\n(6 ML Models × 3 Targets)', orange, 'Ch 4'),
    ('Cross-Validation\n(5-Fold Time-Series)', orange, 'Ch 4'),
    ('Model Evaluation\n(RMSE, MAE, MAPE, R²)', red, 'Ch 5'),
    ('Anomaly Detection\n(Precision, Recall, F1)', red, 'Ch 5'),
    ('SHAP Explainability\n(Feature Importance)', purple, 'Ch 5'),
    ('Results & Visualisation', purple, 'Ch 5'),
    ('Conclusions &\nRecommendations', grey, 'Ch 6'),
]

start_y = 21.5
for i, (text, color, ch) in enumerate(steps):
    y = start_y - i*(bh+gap)
    draw_box(ax, bx, y, bw, bh, text, color)
    ax.text(bx-0.3, y+bh/2, ch, ha='right', va='center', fontsize=11, fontstyle='italic', color='#777')
    if i < len(steps)-1:
        draw_arrow(ax, cx, y, y-gap)

legend_items = [mpatches.Patch(color=c, label=l) for c,l in
    [(grey,'Research Framing'),(blue,'Data Pipeline'),(green,'Feature Engineering'),
     (orange,'Model Development'),(red,'Evaluation'),(purple,'Interpretation')]]
ax.legend(handles=legend_items, loc='lower right', fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.savefig('viz_research_workflow.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: viz_research_workflow.png")

print(f"\n{'='*60}")
print("ALL VISUALISATIONS COMPLETE!")
print(f"{'='*60}")