# ---
# Converted from generate.ipynb
# ---

# ============================================================
# # MLB Bat Tracking Data Generator (2024-2025)
# 
# 2024年シーズンからMLBが導入した新機能「Bat Tracking」のリーダーボードデータを取得します。
# 
# - `statcast_batter_bat_tracking()`: バットスピード、スイング長、squared-up%等の新指標
# ============================================================

# Install pybaseball if needed
# !pip install pybaseball

import pandas as pd
from pybaseball import statcast_batter_bat_tracking
from datetime import date

print(f"Data generation started: {date.today()}")

# ============================================================
# ## 2024シーズン
# ============================================================

# 2024 Regular Season
df_2024 = statcast_batter_bat_tracking(2024, minSwings=50)
df_2024['season'] = 2024
print(f"2024 season: {len(df_2024)} batters")
df_2024.head()

# ============================================================
# ## 2025シーズン（途中まで）
# ============================================================

# 2025 Regular Season (partial, if available)
try:
    df_2025 = statcast_batter_bat_tracking(2025, minSwings=50)
    df_2025['season'] = 2025
    print(f"2025 season: {len(df_2025)} batters")
    df_2025.head()
except Exception as e:
    print(f"2025 data not available yet: {e}")
    df_2025 = pd.DataFrame()

# ============================================================
# ## データ結合とエクスポート
# ============================================================

# Combine both seasons
df_combined = pd.concat([df_2024, df_2025], ignore_index=True)
print(f"\nTotal batters: {len(df_combined)}")
print(f"Columns: {len(df_combined.columns)}")
print(f"\nColumn names:")
print(df_combined.columns.tolist())
print(f"\nData types:")
print(df_combined.dtypes)
print(f"\nFirst few rows:")
df_combined.head(10)

# Export to CSV
df_combined.to_csv('mlb_bat_tracking_2024_2025.csv', index=False)
print(f"\nExported to mlb_bat_tracking_2024_2025.csv")
print(f"File size: {len(df_combined)} rows x {len(df_combined.columns)} columns")

# ============================================================
# ## データ概要
# ============================================================

# Summary statistics
print("\n=== Bat Tracking Metrics Summary ===")
numeric_cols = df_combined.select_dtypes(include=['float64', 'int64']).columns
df_combined[numeric_cols].describe()
