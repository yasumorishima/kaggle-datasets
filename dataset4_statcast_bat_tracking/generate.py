# ---
# Converted from generate.ipynb
# ---

# ============================================================
# # MLB Statcast + Bat Tracking Dataset (2024-2025)
# 
# Generate full pitch-by-pitch Statcast data with Bat Tracking metrics for 2024-2025 seasons.
# 
# **Output:** `statcast_bat_tracking_2024_2025.csv` (~2.4GB, ~1.4M rows)
# ============================================================

# Install required packages
# !pip install pybaseball -q  # uncomment in Colab/notebook

import pandas as pd
import numpy as np
from pybaseball import statcast
from datetime import date

# ============================================================
# ## Fetch 2024 Season
# ============================================================

# 2024 regular season: March 20 - September 29
print('=== Fetching 2024 Season ===')
df_2024 = statcast(start_dt='2024-03-20', end_dt='2024-09-29')
print(f'2024 rows: {len(df_2024):,}')
print(f'2024 bat tracking: {df_2024["bat_speed"].notna().sum():,}')

# ============================================================
# ## Fetch 2025 Season
# ============================================================

# 2025 regular season: March 27 - September 28
print('\n=== Fetching 2025 Season ===')
df_2025 = statcast(start_dt='2025-03-27', end_dt='2025-09-28')
print(f'2025 rows: {len(df_2025):,}')
print(f'2025 bat tracking: {df_2025["bat_speed"].notna().sum():,}')

# ============================================================
# ## Combine and Save
# ============================================================

# Combine both seasons
df = pd.concat([df_2024, df_2025], ignore_index=True)
print(f'\n=== Combined Dataset ===')
print(f'Total rows: {len(df):,}')
print(f'Total columns: {len(df.columns)}')
print(f'Bat tracking rows: {df["bat_speed"].notna().sum():,}')
print(f'Bat tracking coverage: {df["bat_speed"].notna().sum() / len(df) * 100:.1f}%')

# Check memory usage
size_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f'\nMemory usage: {size_mb:.1f} MB')

# Save to CSV
output_file = 'statcast_bat_tracking_2024_2025.csv'
print(f'\nSaving to {output_file}...')
df.to_csv(output_file, index=False)
print('Done!')
print(f'\nDownload the file and upload to Kaggle.')

# Sample data preview
print('\n=== Sample Data (with bat tracking) ===')
sample_cols = ['game_date', 'pitcher', 'batter', 'player_name', 'events',
               'bat_speed', 'swing_length', 'launch_speed', 'launch_angle',
               'release_speed', 'pitch_type', 'pfx_x', 'pfx_z']
sample = df[df['bat_speed'].notna()][sample_cols].head(20)
print(sample)
