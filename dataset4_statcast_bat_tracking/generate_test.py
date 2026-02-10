# ---
# Converted from generate_test.ipynb
# ---

# ============================================================
# # MLB Statcast + Bat Tracking Dataset (2024-2025)
# 
# Generate pitch-by-pitch Statcast data with Bat Tracking metrics.
# 
# ## Test Version (1 week)
# ============================================================

# Install required packages
# !pip install pybaseball -q  # uncomment in Colab/notebook

import pandas as pd
import numpy as np
from pybaseball import statcast
from datetime import date, timedelta

# Test: 1 week of 2024 season
start_date = '2024-09-01'
end_date = '2024-09-07'

print(f'Fetching data: {start_date} to {end_date}')
df = statcast(start_dt=start_date, end_dt=end_date)
print(f'Total rows: {len(df):,}')
print(f'Total columns: {len(df.columns)}')

# Check Bat Tracking coverage
bat_tracking_cols = ['bat_speed', 'swing_length', 'swing_path_tilt']
print('\n=== Bat Tracking Coverage ===')
for col in bat_tracking_cols:
    non_null = df[col].notna().sum()
    pct = non_null / len(df) * 100
    print(f'{col}: {non_null:,} ({pct:.1f}%)')

# Check data types and memory usage
print('\n=== Memory Usage ===')
print(f'Total memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
print(f'\nPer 1000 rows: {df.memory_usage(deep=True).sum() / len(df) * 1000 / 1024**2:.2f} MB')

# Sample data with bat tracking
print('\n=== Sample Data ===')
sample_cols = ['game_date', 'batter', 'player_name', 'events',
               'bat_speed', 'swing_length', 'launch_speed', 'launch_angle']
sample = df[df['bat_speed'].notna()][sample_cols].head(10)
print(sample)

# Estimate full dataset size
# 2024 season: ~183 days, 2025 season: ~183 days (total ~366 days)
days_test = 7
days_full = 366
rows_full = len(df) * days_full / days_test
size_full_mb = df.memory_usage(deep=True).sum() / 1024**2 * days_full / days_test

print('\n=== Full Dataset Estimate (2024-2025) ===')
print(f'Estimated rows: {rows_full:,.0f}')
print(f'Estimated size: {size_full_mb:.0f} MB')
print(f'Bat tracking rows: {df["bat_speed"].notna().sum() * days_full / days_test:,.0f}')
