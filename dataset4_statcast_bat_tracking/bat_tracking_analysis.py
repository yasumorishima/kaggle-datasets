# Auto-extracted from bat_tracking_analysis.ipynb

# %% Cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# %% Cell 2
# Load dataset from Kaggle
df = pd.read_csv('/kaggle/input/mlb-statcast-bat-tracking-2024-2025/statcast_bat_tracking_2024_2025.csv')

print(f'Total rows: {len(df):,}')
print(f'Total columns: {len(df.columns)}')
print(f'\nFirst few rows:')
df.head()

# %% Cell 3
# Check bat tracking data availability
bat_tracking_cols = ['bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction']

print('Bat Tracking Data Coverage:')
print('='*50)
for col in bat_tracking_cols:
    count = df[col].notna().sum()
    pct = count / len(df) * 100
    print(f'{col:25s}: {count:7,} ({pct:5.2f}%)')

# Overall bat tracking coverage (any metric available)
bat_tracking_available = df[bat_tracking_cols].notna().any(axis=1).sum()
print(f'\nTotal pitches with bat tracking: {bat_tracking_available:,} ({bat_tracking_available/len(df)*100:.2f}%)')

# %% Cell 4
# Filter data with bat_speed available
df_bat = df[df['bat_speed'].notna()].copy()

print('Bat Speed Statistics (mph):')
print('='*50)
print(df_bat['bat_speed'].describe())

# Distribution plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(df_bat['bat_speed'], bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(df_bat['bat_speed'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_bat["bat_speed"].mean():.1f} mph')
axes[0].axvline(df_bat['bat_speed'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_bat["bat_speed"].median():.1f} mph')
axes[0].set_xlabel('Bat Speed (mph)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Bat Speed Distribution')
axes[0].legend()

# Box plot
axes[1].boxplot(df_bat['bat_speed'], vert=True)
axes[1].set_ylabel('Bat Speed (mph)')
axes[1].set_title('Bat Speed Box Plot')
axes[1].set_xticklabels([''])

plt.tight_layout()
plt.show()

# %% Cell 5
# Swing length statistics
print('Swing Length Statistics (feet):')
print('='*50)
print(df_bat['swing_length'].describe())

# Distribution plot
plt.figure(figsize=(12, 5))
plt.hist(df_bat['swing_length'].dropna(), bins=50, edgecolor='black', alpha=0.7)
plt.axvline(df_bat['swing_length'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_bat["swing_length"].mean():.2f} ft')
plt.axvline(df_bat['swing_length'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_bat["swing_length"].median():.2f} ft')
plt.xlabel('Swing Length (feet)')
plt.ylabel('Frequency')
plt.title('Swing Length Distribution')
plt.legend()
plt.show()

# %% Cell 6
# Swing path tilt statistics
print('Swing Path Tilt Statistics (degrees):')
print('='*50)
print(df_bat['swing_path_tilt'].describe())

# Distribution plot
plt.figure(figsize=(12, 5))
plt.hist(df_bat['swing_path_tilt'].dropna(), bins=50, edgecolor='black', alpha=0.7)
plt.axvline(df_bat['swing_path_tilt'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_bat["swing_path_tilt"].mean():.1f}°')
plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Level swing (0°)')
plt.xlabel('Swing Path Tilt (degrees)')
plt.ylabel('Frequency')
plt.title('Swing Path Tilt Distribution (Positive = Uppercut, Negative = Downward)')
plt.legend()
plt.show()

# %% Cell 7
# Filter data with both bat_speed and launch_speed
df_corr = df_bat[df_bat['launch_speed'].notna()].copy()

print(f'Data points with both bat_speed and launch_speed: {len(df_corr):,}')
print(f'\nCorrelation coefficient: {df_corr[["bat_speed", "launch_speed"]].corr().iloc[0, 1]:.3f}')

# Scatter plot
plt.figure(figsize=(12, 8))
plt.hexbin(df_corr['bat_speed'], df_corr['launch_speed'], gridsize=50, cmap='YlOrRd', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('Bat Speed (mph)')
plt.ylabel('Launch Speed (Exit Velocity, mph)')
plt.title('Bat Speed vs Launch Speed')

# Add regression line
z = np.polyfit(df_corr['bat_speed'], df_corr['launch_speed'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_corr['bat_speed'].min(), df_corr['bat_speed'].max(), 100)
plt.plot(x_line, p(x_line), 'b--', linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
plt.legend()
plt.show()

# %% Cell 8
# Filter data with both bat_speed and swing_length
df_corr2 = df_bat[df_bat['swing_length'].notna()].copy()

print(f'Correlation coefficient: {df_corr2[["bat_speed", "swing_length"]].corr().iloc[0, 1]:.3f}')

# Scatter plot
plt.figure(figsize=(12, 8))
plt.hexbin(df_corr2['swing_length'], df_corr2['bat_speed'], gridsize=50, cmap='viridis', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('Swing Length (feet)')
plt.ylabel('Bat Speed (mph)')
plt.title('Swing Length vs Bat Speed')

# Add regression line
z = np.polyfit(df_corr2['swing_length'], df_corr2['bat_speed'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_corr2['swing_length'].min(), df_corr2['swing_length'].max(), 100)
plt.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
plt.legend()
plt.show()

# %% Cell 9
# Calculate average bat speed by player (minimum 100 swings)
player_bat_speed = df_bat.groupby('player_name').agg({
    'bat_speed': ['mean', 'count']
}).reset_index()
player_bat_speed.columns = ['player_name', 'avg_bat_speed', 'swings']
player_bat_speed = player_bat_speed[player_bat_speed['swings'] >= 100].sort_values('avg_bat_speed', ascending=False)

print('Top 10 Bat Speed Leaders (min. 100 swings):')
print('='*70)
print(player_bat_speed.head(10).to_string(index=False))

# Bar plot
top10 = player_bat_speed.head(10)
plt.figure(figsize=(12, 6))
plt.barh(range(len(top10)), top10['avg_bat_speed'], color='steelblue')
plt.yticks(range(len(top10)), top10['player_name'])
plt.xlabel('Average Bat Speed (mph)')
plt.title('Top 10 Bat Speed Leaders (2024-2025)')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# %% Cell 10
# Average bat speed by pitch type
pitch_type_bat_speed = df_bat.groupby('pitch_type').agg({
    'bat_speed': ['mean', 'count']
}).reset_index()
pitch_type_bat_speed.columns = ['pitch_type', 'avg_bat_speed', 'count']
pitch_type_bat_speed = pitch_type_bat_speed[pitch_type_bat_speed['count'] >= 100].sort_values('avg_bat_speed', ascending=False)

print('Average Bat Speed by Pitch Type (min. 100 swings):')
print('='*60)
print(pitch_type_bat_speed.to_string(index=False))

# Bar plot
plt.figure(figsize=(12, 6))
plt.bar(pitch_type_bat_speed['pitch_type'], pitch_type_bat_speed['avg_bat_speed'], color='coral')
plt.xlabel('Pitch Type')
plt.ylabel('Average Bat Speed (mph)')
plt.title('Average Bat Speed by Pitch Type')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% Cell 11
# Filter data with attack_angle and attack_direction
df_attack = df_bat[df_bat['attack_angle'].notna() & df_bat['attack_direction'].notna()].copy()

# Create bins
df_attack['attack_angle_bin'] = pd.cut(df_attack['attack_angle'], bins=20)
df_attack['attack_direction_bin'] = pd.cut(df_attack['attack_direction'], bins=20)

# Pivot table
heatmap_data = df_attack.pivot_table(
    values='bat_speed',
    index='attack_angle_bin',
    columns='attack_direction_bin',
    aggfunc='mean'
)

# Plot
plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, cmap='RdYlGn', cbar_kws={'label': 'Bat Speed (mph)'})
plt.xlabel('Attack Direction (degrees)')
plt.ylabel('Attack Angle (degrees)')
plt.title('Average Bat Speed Heatmap (Attack Angle vs Attack Direction)')
plt.tight_layout()
plt.show()
