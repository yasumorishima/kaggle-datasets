# ---
# Converted from pitcher_arsenal_evolution_2020_2025.ipynb
# ---

# ============================================================
# # MLB Pitcher Arsenal Evolution (2020-2025)
# 
# A dataset tracking the yearly evolution of pitcher arsenals (pitch mix composition)
# 
# **Data Format**: Wide format (1 row = pitcher × season, pitch types expanded horizontally)
# 
# **Coverage Period**: 2020-2025 seasons (6 seasons)
# 
# **Target Pitchers**: Pitchers who threw 100+ pitches per season
# 
# **Main Pitch Types**: FF, SI, FC, SL, CU, CH, FS, KC, FO, EP, KN
# 
# **Metrics (per pitch type)**:
# - usage_pct: Usage percentage (%)
# - avg_speed: Average velocity (mph)
# - avg_spin: Average spin rate (rpm)
# - whiff_rate: Whiff rate
# - avg_pfx_x: Average horizontal movement (inches)
# - avg_pfx_z: Average vertical movement (inches)
# ============================================================

# Install required packages
# !pip install -q pybaseball duckdb  # uncomment in Colab/notebook

import pandas as pd
import numpy as np
from datetime import date
from pybaseball import statcast, playerid_reverse_lookup
import duckdb
import warnings
warnings.filterwarnings('ignore')

print(f"Data collection date: {date.today()}")

# ============================================================
# ## Step 1: Data Collection (2020-2025)
# 
# **Note**: This process may take 30-60 minutes
# ============================================================

# Fetch data for each season
seasons = [2020, 2021, 2022, 2023, 2024, 2025]
all_data = []

for season in seasons:
    print(f"\nFetching {season} season data...")
    start_date = f"{season}-03-01"
    end_date = f"{season}-11-30" if season < 2025 else date.today().strftime("%Y-%m-%d")
    
    df = statcast(start_dt=start_date, end_dt=end_date)
    df['season'] = season
    all_data.append(df)
    print(f"  {season}: {len(df):,} pitches")

# Concatenate all seasons
df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal pitches: {len(df_all):,}")
print(f"Columns: {len(df_all.columns)}")

# ============================================================
# ## Step 2: Aggregate with DuckDB
# 
# Group by pitcher × season × pitch type and calculate statistics
# ============================================================

con = duckdb.connect()

# Aggregate by pitcher × season × pitch type
query = """
WITH pitcher_stats AS (
    SELECT
        pitcher,
        season,
        pitch_type,
        COUNT(*) as n_pitches,
        AVG(release_speed) as avg_speed,
        AVG(release_spin_rate) as avg_spin,
        AVG(pfx_x) as avg_pfx_x,
        AVG(pfx_z) as avg_pfx_z,
        SUM(CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked') THEN 1 ELSE 0 END)::FLOAT /
            NULLIF(SUM(CASE WHEN description LIKE '%strike%' OR description LIKE '%foul%' OR description IN ('hit_into_play') THEN 1 ELSE 0 END), 0) as whiff_rate
    FROM df_all
    WHERE pitch_type IS NOT NULL
        AND pitcher IS NOT NULL
    GROUP BY pitcher, season, pitch_type
),
pitcher_totals AS (
    SELECT
        pitcher,
        season,
        SUM(n_pitches) as total_pitches
    FROM pitcher_stats
    GROUP BY pitcher, season
    HAVING total_pitches >= 100  -- Minimum 100 pitches
)
SELECT
    ps.pitcher,
    ps.season,
    ps.pitch_type,
    ps.n_pitches,
    ROUND(100.0 * ps.n_pitches / pt.total_pitches, 2) as usage_pct,
    ROUND(ps.avg_speed, 2) as avg_speed,
    ROUND(ps.avg_spin, 0) as avg_spin,
    ROUND(ps.whiff_rate, 4) as whiff_rate,
    ROUND(ps.avg_pfx_x, 2) as avg_pfx_x,
    ROUND(ps.avg_pfx_z, 2) as avg_pfx_z
FROM pitcher_stats ps
INNER JOIN pitcher_totals pt
    ON ps.pitcher = pt.pitcher AND ps.season = pt.season
ORDER BY ps.pitcher, ps.season, ps.pitch_type
"""

df_long = con.execute(query).df()
print(f"Long format: {len(df_long):,} rows (pitcher × season × pitch type)")
print(f"Unique pitchers: {df_long['pitcher'].nunique():,}")
print(f"\nPitch types: {sorted(df_long['pitch_type'].unique())}")

# Sample preview
df_long.head(20)

# ============================================================
# ## Step 3: Convert to Wide Format
# 
# Expand pitch types horizontally (FF_usage_pct, FF_avg_speed, ...)
# ============================================================

# Metrics list
metrics = ['usage_pct', 'avg_speed', 'avg_spin', 'whiff_rate', 'avg_pfx_x', 'avg_pfx_z']

# Pivot for each metric
pivoted_dfs = []

for metric in metrics:
    pivot = df_long.pivot_table(
        index=['pitcher', 'season'],
        columns='pitch_type',
        values=metric,
        aggfunc='first'
    )
    # Rename columns to "PITCH_metric" format
    pivot.columns = [f"{col}_{metric}" for col in pivot.columns]
    pivoted_dfs.append(pivot)

# Concatenate all metrics
df_wide = pd.concat(pivoted_dfs, axis=1).reset_index()

print(f"Wide format: {len(df_wide):,} rows (pitcher × season)")
print(f"Columns: {len(df_wide.columns)}")

# List all columns
print("\nColumn names:")
for i, col in enumerate(df_wide.columns, 1):
    print(f"{i:3d}. {col}")

# ============================================================
# ## Step 4: Add Pitcher Names
# ============================================================

# Get unique pitcher IDs
unique_pitchers = df_wide['pitcher'].unique()
print(f"Looking up names for {len(unique_pitchers):,} pitchers...")

# Fetch pitcher names (batch processing)
name_dict = {}
batch_size = 100

for i in range(0, len(unique_pitchers), batch_size):
    batch = unique_pitchers[i:i+batch_size]
    for player_id in batch:
        try:
            result = playerid_reverse_lookup([player_id], key_type='mlbam')
            if not result.empty:
                name_dict[player_id] = f"{result.iloc[0]['name_first']} {result.iloc[0]['name_last']}"
        except:
            name_dict[player_id] = f"Player_{player_id}"
    
    if (i + batch_size) % 500 == 0:
        print(f"  {i + batch_size:,} / {len(unique_pitchers):,}")

# Add pitcher names
df_wide.insert(1, 'player_name', df_wide['pitcher'].map(name_dict))

print(f"\nCompleted. {df_wide['player_name'].notna().sum()} names found.")

# Sample preview
df_wide.head(10)

# ============================================================
# ## Step 5: Export to CSV
# ============================================================

# Rename column from pitcher to player_id
df_wide.rename(columns={'pitcher': 'player_id'}, inplace=True)

# Export to CSV
output_file = "pitcher_arsenal_evolution_2020_2025.csv"
df_wide.to_csv(output_file, index=False)

print(f"\n=== Dataset Summary ===")
print(f"File: {output_file}")
print(f"Rows: {len(df_wide):,}")
print(f"Columns: {len(df_wide.columns)}")
print(f"Pitchers: {df_wide['player_id'].nunique():,}")
print(f"Seasons: {sorted(df_wide['season'].unique())}")
print(f"\nFile size: {df_wide.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB (in memory)")

# Basic statistics
print("\n=== Basic Statistics ===")
print(f"\nPitchers per season:")
print(df_wide.groupby('season')['player_id'].nunique())

print(f"\nMost common pitch types (by usage):")
usage_cols = [col for col in df_wide.columns if col.endswith('_usage_pct')]
for col in usage_cols:
    mean_usage = df_wide[col].mean()
    if pd.notna(mean_usage) and mean_usage > 1.0:  # Only pitch types with 1%+ usage
        pitch_type = col.replace('_usage_pct', '')
        print(f"  {pitch_type}: {mean_usage:.2f}%")
