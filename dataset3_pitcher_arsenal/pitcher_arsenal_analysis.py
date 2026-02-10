# ---
# Converted from pitcher_arsenal_analysis.ipynb
# ---

# ============================================================
# # MLB Pitcher Arsenal Evolution Analysis (2020-2025)
# 
# This notebook analyzes pitcher arsenal changes using the [Pitcher Arsenal Evolution Dataset](https://www.kaggle.com/datasets/yasunorim/pitcher-arsenal-evolution-2020-2025).
# 
# ## 📊 Analysis Contents
# 1. Dataset Overview
# 2. Individual Pitcher Trend Analysis (Example: Yusei Kikuchi's slider increase)
# 3. MLB-Wide Pitch Type Trends (2020-2025)
# 4. Pitch Velocity Changes
# 5. Use Case Examples
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Japanese font setup (for Google Colab)
# !pip install -q japanize-matplotlib  # uncomment in Colab/notebook
import japanize_matplotlib

# Plot settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# ============================================================
# ## 1. Load Data
# ============================================================

# Load from Kaggle dataset
df = pd.read_csv('/kaggle/input/pitcher-arsenal-evolution-2020-2025/pitcher_arsenal_evolution_2020_2025.csv')

print(f"Dataset shape: {df.shape}")
print(f"Pitchers: {df['player_id'].nunique():,}")
print(f"Seasons: {sorted(df['season'].unique())}")
print(f"\nColumns: {len(df.columns)}")

# View first 5 rows
df.head()

# Column list
print("\nAll columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

# ============================================================
# ## 2. Basic Statistics
# ============================================================

# Pitchers per season
print("Pitchers per season:")
print(df.groupby('season')['player_id'].nunique().sort_index())

# Check missing values
print("\nMissing values by column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False).head(20))

# Average usage of main pitch types
usage_cols = [col for col in df.columns if col.endswith('_usage_pct')]

print("\nAverage pitch usage across all pitchers:")
usage_means = df[usage_cols].mean().sort_values(ascending=False)
for col, val in usage_means.items():
    if pd.notna(val) and val > 1.0:
        pitch_type = col.replace('_usage_pct', '')
        print(f"  {pitch_type:4s}: {val:5.2f}%")

# ============================================================
# ## 3. Individual Pitcher Trend Analysis
# 
# ### Example: Yusei Kikuchi's Slider Increase
# ============================================================

# Extract specific pitcher data (Example: Yusei Kikuchi)
# Search by partial player_name match
pitcher_name = "Yusei Kikuchi"
df_pitcher = df[df['player_name'].str.contains(pitcher_name, case=False, na=False)].sort_values('season')

if len(df_pitcher) > 0:
    print(f"\n{pitcher_name} - Seasons found: {len(df_pitcher)}")
    print(df_pitcher[['season', 'player_name']].to_string(index=False))
else:
    print(f"\n{pitcher_name} not found. Searching for similar names...")
    # Search for similar names
    similar = df[df['player_name'].str.contains('Kikuchi', case=False, na=False)]['player_name'].unique()
    print(f"Found: {similar}")

# Plot pitch usage trends
if len(df_pitcher) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot main pitch types
    pitch_types = ['FF', 'SI', 'SL', 'CU', 'CH', 'FC']
    for pitch in pitch_types:
        col = f"{pitch}_usage_pct"
        if col in df_pitcher.columns:
            usage = df_pitcher[col].values
            if not all(pd.isna(usage)):
                ax.plot(df_pitcher['season'], usage, marker='o', label=pitch, linewidth=2)
    
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Usage (%)', fontsize=12)
    ax.set_title(f'{pitcher_name} - Pitch Usage Evolution (2020-2025)', fontsize=14, fontweight='bold')
    ax.legend(title='Pitch Type', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# ## 4. MLB-Wide Pitch Type Trends
# ============================================================

# Calculate average usage per season
pitch_types_main = ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'FS']

trend_data = []
for season in sorted(df['season'].unique()):
    df_season = df[df['season'] == season]
    for pitch in pitch_types_main:
        col = f"{pitch}_usage_pct"
        if col in df.columns:
            mean_usage = df_season[col].mean()
            if pd.notna(mean_usage):
                trend_data.append({
                    'season': season,
                    'pitch_type': pitch,
                    'avg_usage': mean_usage
                })

df_trend = pd.DataFrame(trend_data)
df_trend.head(10)

# Plot trends
fig, ax = plt.subplots(figsize=(14, 7))

for pitch in pitch_types_main:
    df_pitch = df_trend[df_trend['pitch_type'] == pitch]
    if len(df_pitch) > 0:
        ax.plot(df_pitch['season'], df_pitch['avg_usage'], marker='o', label=pitch, linewidth=2.5)

ax.set_xlabel('Season', fontsize=13)
ax.set_ylabel('Average Usage (%)', fontsize=13)
ax.set_title('MLB Pitch Type Trends (2020-2025)', fontsize=15, fontweight='bold')
ax.legend(title='Pitch Type', fontsize=11, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ## 5. Pitch Velocity Analysis
# ============================================================

# Average velocity trends by pitch type
speed_data = []
for season in sorted(df['season'].unique()):
    df_season = df[df['season'] == season]
    for pitch in pitch_types_main:
        col = f"{pitch}_avg_speed"
        if col in df.columns:
            mean_speed = df_season[col].mean()
            if pd.notna(mean_speed):
                speed_data.append({
                    'season': season,
                    'pitch_type': pitch,
                    'avg_speed': mean_speed
                })

df_speed = pd.DataFrame(speed_data)

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

for pitch in pitch_types_main:
    df_pitch = df_speed[df_speed['pitch_type'] == pitch]
    if len(df_pitch) > 0:
        ax.plot(df_pitch['season'], df_pitch['avg_speed'], marker='s', label=pitch, linewidth=2.5)

ax.set_xlabel('Season', fontsize=13)
ax.set_ylabel('Average Speed (mph)', fontsize=13)
ax.set_title('MLB Pitch Speed Trends by Type (2020-2025)', fontsize=15, fontweight='bold')
ax.legend(title='Pitch Type', fontsize=11, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ## 6. Heatmap: Pitcher × Pitch Type
# ============================================================

# Create heatmap for top 20 pitchers in 2025 season
df_2025 = df[df['season'] == 2025].copy()

# Sort pitchers by total usage (sum of main pitch type usage as proxy)
usage_sum = df_2025[usage_cols].sum(axis=1)
df_2025['total_usage'] = usage_sum
df_top20 = df_2025.nlargest(20, 'total_usage')

# Prepare heatmap data
heatmap_data = df_top20[['player_name'] + [f"{p}_usage_pct" for p in pitch_types_main]].set_index('player_name')
heatmap_data.columns = [col.replace('_usage_pct', '') for col in heatmap_data.columns]

# Plot
fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Usage (%)'}, ax=ax)
ax.set_title('Top 20 Pitchers - Pitch Usage Heatmap (2025)', fontsize=14, fontweight='bold')
ax.set_xlabel('Pitch Type', fontsize=12)
ax.set_ylabel('Pitcher', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# ## 7. Use Case Example: Pre/Post-Injury Changes
# 
# Analyze arsenal changes when a pitcher returns from injury
# ============================================================

# Example: Changes across 2023→2024→2025 for a specific pitcher
example_pitcher = "Jacob deGrom"  # Example
df_example = df[df['player_name'].str.contains(example_pitcher, case=False, na=False)].sort_values('season')

if len(df_example) >= 2:
    print(f"\n{example_pitcher} - Arsenal Changes:")
    
    for pitch in pitch_types_main:
        col_usage = f"{pitch}_usage_pct"
        col_speed = f"{pitch}_avg_speed"
        
        if col_usage in df_example.columns:
            usage_values = df_example[col_usage].values
            speed_values = df_example[col_speed].values if col_speed in df_example.columns else []
            
            if not all(pd.isna(usage_values)):
                print(f"\n{pitch}:")
                for i, season in enumerate(df_example['season'].values):
                    usage = usage_values[i]
                    speed = speed_values[i] if len(speed_values) > i else np.nan
                    if pd.notna(usage):
                        speed_str = f", {speed:.1f} mph" if pd.notna(speed) else ""
                        print(f"  {season}: {usage:.1f}%{speed_str}")
else:
    print(f"{example_pitcher} - Not enough data for comparison")

# ============================================================
# ## 8. Summary
# 
# This dataset enables the following analyses:
# - Track individual pitcher arsenal evolution
# - Analyze MLB-wide trends
# - Detect changes before/after injury or trades
# - Compare team strategies
# - Feature engineering for machine learning models
# 
# ### Next Steps
# - More detailed statistical analysis (t-tests, ANOVA, etc.)
# - Machine learning models (performance prediction, etc.)
# - Create interactive dashboards
# ============================================================
