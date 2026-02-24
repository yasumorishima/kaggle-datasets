# Auto-extracted from wbc_spraychart.ipynb

# %% Cell 1
!pip install baseball-field-viz -q

import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from baseball_field_viz import transform_coords, draw_field, spraychart, draw_strike_zone, pitch_zone_chart

print("baseball-field-viz ready")

# %% Cell 2
def _find_data_dir():
    candidates = [
        "/kaggle/input/wbc-2026-scouting",
        "/kaggle/input/datasets/yasunorim/wbc-2026-scouting",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Data not found. Tried: {candidates}")

DATA_DIR = _find_data_dir()
print(f"Data directory: {DATA_DIR}")

df = pd.read_csv(f"{DATA_DIR}/statcast_batters.csv")
rosters = pd.read_csv(f"{DATA_DIR}/rosters.csv")

print(f"Statcast records : {len(df):,}")
print(f"Countries        : {df['country_name'].nunique()}")
print(f"Players          : {df['player_name'].nunique()}")
print(f"Batted balls     : {df['hc_x'].notna().sum():,}")
print()

hit_events = ['home_run', 'double', 'triple', 'single']
hits_by_country = (
    df[df['events'].isin(hit_events)]
    .groupby('country_name')
    .size()
    .sort_values(ascending=False)
)
print("Hits by country:")
print(hits_by_country.to_string())

# %% Cell 3
hits = df[
    df['hc_x'].notna() & df['events'].isin(hit_events)
].copy()
hits = transform_coords(hits)

country_list = sorted(hits['country_name'].unique())
colors = cm.tab20(np.linspace(0, 1, len(country_list)))
color_map = dict(zip(country_list, colors))

fig, ax = plt.subplots(figsize=(12, 12))
draw_field(ax)

for country in country_list:
    subset = hits[hits['country_name'] == country]
    ax.scatter(
        subset['x'], subset['y'],
        c=[color_map[country]], alpha=0.35, s=12,
        label=f"{country} ({len(subset)})"
    )

ax.set_xlim(-350, 350)
ax.set_ylim(-50, 420)
ax.set_xlabel("X (feet)")
ax.set_ylabel("Y (feet)")
ax.set_title("WBC 2026 — All Countries Batted Balls (Hits Only)", fontsize=14)
ax.legend(loc='upper right', fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# %% Cell 4
top_countries = ['USA', 'Dominican Republic', 'Venezuela', 'Japan']

fig, axs = plt.subplots(2, 2, figsize=(16, 14))

for ax, country in zip(axs.flat, top_countries):
    df_c = df[df['country_name'] == country]
    df_ct = transform_coords(df_c[df_c['hc_x'].notna()])

    draw_field(ax)
    sns.kdeplot(
        data=df_ct, x='x', y='y', ax=ax,
        fill=True, alpha=0.6, cmap='Reds', levels=8,
        clip=((-350, 350), (0, 400)),
        bw_adjust=0.7,
        thresh=0.1,
    )
    ax.set_xlim(-350, 350)
    ax.set_ylim(-50, 400)
    ax.set_title(f"{country}  (n={len(df_ct):,})", fontsize=13)

plt.suptitle("WBC 2026 — Batted Ball Density by Country", fontsize=16, y=1.01)
plt.tight_layout()
plt.show()

# %% Cell 5
df_jpn = df[df['country_name'] == 'Japan']
df_jpn_t = transform_coords(df_jpn[df_jpn['hc_x'].notna()])

hits_jpn = df_jpn_t[df_jpn_t['events'].isin(hit_events)]
outs_jpn = df_jpn_t[~df_jpn_t['events'].isin(hit_events)]

fig, axs = plt.subplots(1, 2, figsize=(16, 8))

draw_field(axs[0])
sns.kdeplot(
    data=hits_jpn, x='x', y='y', ax=axs[0],
    cmap='Reds', fill=True, alpha=0.6, levels=8,
    clip=((-350, 350), (0, 400)), bw_adjust=0.7, thresh=0.1,
)
axs[0].set_xlim(-350, 350)
axs[0].set_ylim(-50, 400)
axs[0].set_title(f"Japan — Hits Heatmap  (n={len(hits_jpn)})", fontsize=13)

draw_field(axs[1])
sns.kdeplot(
    data=outs_jpn, x='x', y='y', ax=axs[1],
    cmap='Blues', fill=True, alpha=0.6, levels=8,
    clip=((-350, 350), (0, 400)), bw_adjust=0.7, thresh=0.1,
)
axs[1].set_xlim(-350, 350)
axs[1].set_ylim(-50, 400)
axs[1].set_title(f"Japan — Outs Heatmap  (n={len(outs_jpn)})", fontsize=13)

plt.suptitle("Japan — Batted Ball Distribution", fontsize=15)
plt.tight_layout()
plt.show()

# %% Cell 6
hr_counts = (
    df[df['events'] == 'home_run']
    .groupby('country_name')
    .size()
    .sort_values(ascending=False)
    .head(5)
)
top5_hr_countries = hr_counts.index.tolist()

fig, axs = plt.subplots(1, 5, figsize=(28, 7))

for ax, country in zip(axs, top5_hr_countries):
    df_c = df[(df['country_name'] == country) & (df['events'] == 'home_run')]
    n = len(df_c[df_c['hc_x'].notna()])
    spraychart(ax, df_c, color_by='events', title=f"{country}\n({n} HR)")
    # remove legend for compact display
    ax.get_legend().remove()

plt.suptitle("WBC 2026 — Home Run Spray Charts (Top 5 Countries)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% Cell 7
# --- draw_strike_zone demo ---
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for ax, (sz_t, sz_b, label) in zip(axs, [
    (3.5, 1.5, "Average MLB Strike Zone"),
    (4.0, 1.3, "Taller Batter (e.g. 6'7\")"),
]):
    draw_strike_zone(ax, sz_top=sz_t, sz_bot=sz_b)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(0, 5.5); ax.set_aspect('equal')
    ax.set_xlabel("plate_x (ft)"); ax.set_ylabel("plate_z (ft)")
    ax.set_title(f"{label}\n(sz_top={sz_t}, sz_bot={sz_b})")
plt.suptitle("draw_strike_zone()", fontsize=14)
plt.tight_layout()
plt.show()

# --- pitch_zone_chart: one subplot per pitch type ---
df_p = pd.read_csv(f"{DATA_DIR}/statcast_pitchers.csv")
jpn_p = df_p[df_p['country_name'] == 'Japan']

if 'plate_x' in jpn_p.columns and 'plate_z' in jpn_p.columns:
    top_pitcher = jpn_p['player_name'].value_counts().index[0]
    df_one = jpn_p[jpn_p['player_name'] == top_pitcher]

    pitch_counts = df_one['pitch_type'].value_counts()
    pitch_types = pitch_counts.index.tolist()
    print(f"Pitcher: {top_pitcher}  (total {len(df_one):,} pitches)")
    print(pitch_counts.to_string())

    ncols = min(len(pitch_types), 3)
    nrows = (len(pitch_types) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axs_flat = axs.flat if hasattr(axs, 'flat') else [axs]

    for ax, pt in zip(axs_flat, pitch_types):
        df_pt = df_one[df_one['pitch_type'] == pt]
        pitch_zone_chart(ax, df_pt, title=f"{pt}  (n={len(df_pt)})")

    for ax in list(axs_flat)[len(pitch_types):]:
        ax.set_visible(False)

    plt.suptitle(f"{top_pitcher} — Pitch Location by Type", fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("Note: plate_x/plate_z not in this dataset.")
    print("Use pybaseball.statcast_pitcher() for full pitch location data.")
