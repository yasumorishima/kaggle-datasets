# %% [markdown]
# # MLB Pitcher Arsenal Analysis (2020-2026)
#
# This notebook uses the dataset **[MLB Pitcher Arsenal 2020-2026](https://www.kaggle.com/datasets/yasunorim/mlb-pitcher-arsenal-2020-2025)**: what every MLB pitcher threw in the 2020-2026 regular seasons, pitch type by pitch type, taken from Baseball Savant's leaderboards.
#
# The dataset was rebuilt in September 2026. The old single file `pitcher_arsenal_evolution_2020_2025.csv` is replaced by three files:
#
# | File | One row per |
# |---|---|
# | `pitcher_arsenal.csv` | pitcher, season, pitch type |
# | `pitcher_arsenal_wide.csv` | pitcher, season (pitch types side by side: `ff_usage_pct`, `sl_whiff_pct`, ...) |
# | `arsenal_changes.csv` | pitcher, season, pitch type, compared with the season before |
#
# Contents
# 1. Load the data
# 2. Basic statistics
# 3. One pitcher over time (Yusei Kikuchi)
# 4. League-wide pitch mix, 2020-2026
# 5. Velocity by pitch type
# 6. Whiff rate by pitch type
# 7. Heatmap: pitcher x pitch type (2026)
# 8. Before and after an injury (Jacob deGrom)
# 9. Biggest changes from 2025 to 2026

# %%
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                     "legend.title_fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12})

PITCH_NAMES = {"FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper",
               "SV": "Slurve", "CU": "Curveball", "CH": "Changeup", "FS": "Splitter",
               "FO": "Forkball", "SC": "Screwball", "KN": "Knuckleball"}
MAIN = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]
COLORS = dict(zip(MAIN, sns.color_palette("tab10", len(MAIN))))

# %% [markdown]
# ## 1. Load the data

# %%
# On Kaggle the dataset is under /kaggle/input/; KAGGLE_INPUT_ROOT lets the notebook run elsewhere.
INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
paths = glob.glob(os.path.join(INPUT_ROOT, "**", "pitcher_arsenal.csv"), recursive=True)
if not paths:
    raise FileNotFoundError("Attach the dataset yasunorim/mlb-pitcher-arsenal-2020-2025 to this notebook.")
DATA_DIR = os.path.dirname(paths[0])

arsenal = pd.read_csv(os.path.join(DATA_DIR, "pitcher_arsenal.csv"))
wide = pd.read_csv(os.path.join(DATA_DIR, "pitcher_arsenal_wide.csv"))
changes = pd.read_csv(os.path.join(DATA_DIR, "arsenal_changes.csv"))

for name, d in [("pitcher_arsenal", arsenal), ("pitcher_arsenal_wide", wide), ("arsenal_changes", changes)]:
    print(f"{name:22s} {d.shape[0]:6,} rows x {d.shape[1]:3d} columns")
print("Seasons:", sorted(arsenal["season"].unique().tolist()))
print("Pitchers:", f"{arsenal['pitcher_id'].nunique():,}")

# %%
arsenal.head()

# %%
print("pitcher_arsenal.csv columns:")
print(", ".join(arsenal.columns))

# %% [markdown]
# ## 2. Basic statistics
#
# Every pitcher is included, down to position players who threw a few pitches. Pitch types that never ended a plate appearance come only with usage, speed, spin and break (`in_arsenal_stats` = False). Filter on `pitches` or `total_pitches` when you want stable rates.

# %%
per_season = pd.DataFrame({
    "pitchers": wide.groupby("season")["pitcher_id"].nunique(),
    "pitchers_100plus": wide[wide["total_pitches"] >= 100].groupby("season")["pitcher_id"].nunique(),
    "pitch_type_rows": arsenal.groupby("season").size(),
    "pitches": arsenal.groupby("season")["pitches"].sum(),
})
per_season

# %%
print("Share of rows with a missing value, pitcher_arsenal.csv (columns with any):")
miss = arsenal.isna().mean().mul(100).round(1)
print(miss[miss > 0].sort_values(ascending=False).to_string())

# %% [markdown]
# ## 3. One pitcher over time: Yusei Kikuchi
#
# Names are written "Last, First". Savant labels sweepers (ST) separately from sliders (SL) back to 2020, so a pitch reclassified between seasons moves from one line to the other.

# %%
pitcher_name = "Kikuchi, Yusei"
kik = wide[wide["player_name"] == pitcher_name].sort_values("season")
cols = ["season", "team", "total_pitches"] + [f"{p.lower()}_usage_pct" for p in MAIN]
kik[cols].set_index("season").dropna(axis=1, how="all")

# %%
fig, ax = plt.subplots(figsize=(12, 6))
for p in MAIN:
    col = f"{p.lower()}_usage_pct"
    if kik[col].max() >= 3:  # skip pitches he threw only a handful of times
        ax.plot(kik["season"], kik[col], marker="o", lw=2.5, color=COLORS[p], label=PITCH_NAMES[p])
ax.set_xlabel("Season")
ax.set_ylabel("Usage (% of his pitches)")
ax.set_title("Yusei Kikuchi: pitch usage by season, 2020-2026")
ax.set_xticks(kik["season"])
ax.set_ylim(0, None)
ax.legend(title="Pitch type", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# His slider went from 16% of his pitches in 2020 to 36% in 2025, then back to 25% in 2026, the season he added a splitter (17%). Pitches under 3% in every season are left off the chart.

# %% [markdown]
# ## 4. League-wide pitch mix, 2020-2026
#
# Share of all regular-season pitches thrown, by pitch type (pitch counts summed over pitchers). Knuckle curves are counted as curveballs.

# %%
total = arsenal.groupby("season")["pitches"].sum()
share = (arsenal.groupby(["season", "pitch_type"])["pitches"].sum()
         .unstack().div(total, axis=0).mul(100))
share[MAIN].round(1)

# %%
fig, ax = plt.subplots(figsize=(13, 7))
for p in MAIN:
    ax.plot(share.index, share[p], marker="o", lw=2.5, color=COLORS[p], label=PITCH_NAMES[p])
ax.set_xlabel("Season")
ax.set_ylabel("Share of all pitches (%)")
ax.set_title("MLB pitch mix by season, 2020-2026")
ax.set_xticks(share.index)
ax.set_ylim(0, None)
ax.legend(title="Pitch type", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Velocity by pitch type
#
# Average speed weighted by the number of pitches.

# %%
spd = arsenal.dropna(subset=["avg_speed"]).copy()
spd["speed_x_pitches"] = spd["avg_speed"] * spd["pitches"]
g = spd.groupby(["season", "pitch_type"])[["speed_x_pitches", "pitches"]].sum()
speed = (g["speed_x_pitches"] / g["pitches"]).unstack()
speed[MAIN].round(1)

# %%
fig, ax = plt.subplots(figsize=(13, 7))
for p in MAIN:
    ax.plot(speed.index, speed[p], marker="s", lw=2.5, color=COLORS[p], label=PITCH_NAMES[p])
ax.set_xlabel("Season")
ax.set_ylabel("Average speed (mph)")
ax.set_title("MLB average pitch speed by type, 2020-2026")
ax.set_xticks(speed.index)
ax.legend(title="Pitch type", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Whiff rate by pitch type
#
# `whiff_pct` is Savant's whiff%: whiffs divided by swings, where Savant counts foul tips as whiffs.
#
# **Note on the old file.** The previous `whiff_rate` column divided swinging strikes by swings *plus called strikes*, so it was far too low (2025 four-seam median 0.132, against Savant's 21.6%). This notebook did not use that column before; any analysis that did should be redone with `whiff_pct`.
#
# Below: the median over pitchers, for pitch types thrown 250+ times in the season.

# %%
MIN_PITCHES = 250
wh = (arsenal[arsenal["pitches"] >= MIN_PITCHES]
      .groupby(["season", "pitch_type"])["whiff_pct"].median().unstack())
wh[MAIN].round(1)

# %%
fig, ax = plt.subplots(figsize=(13, 7))
for p in MAIN:
    ax.plot(wh.index, wh[p], marker="o", lw=2.5, color=COLORS[p], label=PITCH_NAMES[p])
ax.set_xlabel("Season")
ax.set_ylabel("Median whiff % (whiffs / swings)")
ax.set_title(f"Whiff % by pitch type (pitch types thrown {MIN_PITCHES}+ times)")
ax.set_xticks(wh.index)
ax.set_ylim(0, None)
ax.legend(title="Pitch type", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Heatmap: pitcher x pitch type (2026)
#
# The 20 pitchers who threw the most pitches in 2026, and how they split them.

# %%
YEAR = 2026
top20 = wide[wide["season"] == YEAR].nlargest(20, "total_pitches")
heat = top20.set_index("player_name")[[f"{p.lower()}_usage_pct" for p in MAIN]]
heat.columns = MAIN

fig, ax = plt.subplots(figsize=(11, 12))
labels = heat.map(lambda v: "" if pd.isna(v) else ("<1" if v < 0.5 else f"{v:.0f}"))
sns.heatmap(heat, annot=labels, fmt="", cmap="YlOrRd", vmin=0, vmax=60,
            annot_kws={"fontsize": 11}, cbar_kws={"label": "Usage (%)"}, ax=ax)
ax.set_title(f"Pitch usage of the 20 pitchers with the most pitches, {YEAR}")
ax.set_xlabel("Pitch type")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Before and after an injury: Jacob deGrom
#
# deGrom had Tommy John surgery in June 2023 and came back at the end of 2024. The pitch counts show the gap.

# %%
dg = arsenal[arsenal["player_name"] == "deGrom, Jacob"]
dg_usage = dg.pivot(index="season", columns="pitch_type", values="usage_pct")
dg_speed = dg.pivot(index="season", columns="pitch_type", values="avg_speed")
summary = pd.concat({"pitches": dg.groupby("season")["total_pitches"].first(),
                     **{f"{p} usage %": dg_usage[p] for p in dg_usage.columns},
                     "FF speed (mph)": dg_speed.get("FF")}, axis=1)
summary

# %%
shown = [c for c in MAIN if c in dg_usage.columns and dg_usage[c].max() >= 3]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for p in shown:
    axes[0].plot(dg_usage.index, dg_usage[p], marker="o", lw=2.5, color=COLORS[p], label=PITCH_NAMES[p])
axes[0].set_title("Jacob deGrom: usage by season")
axes[0].set_xlabel("Season")
axes[0].set_ylabel("Usage (%)")
axes[0].set_ylim(0, None)
for p in shown:
    axes[1].plot(dg_speed.index, dg_speed[p], marker="s", lw=2.5, color=COLORS[p], label=PITCH_NAMES[p])
axes[1].set_title("Jacob deGrom: average speed by season")
axes[1].set_xlabel("Season")
axes[1].set_ylabel("Average speed (mph)")
for ax in axes:
    ax.set_xticks(dg_usage.index)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Pitch type", loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Biggest changes from 2025 to 2026
#
# `arsenal_changes.csv` has one row per pitcher, season and pitch type, with the previous season's value next to the current one. Here: the largest usage increases among pitchers who threw 1,000+ pitches in both seasons.

# %%
workload = wide.set_index(["pitcher_id", "season"])["total_pitches"]
ch = changes[changes["season"] == 2026].copy()
ch["pitches_now"] = workload.reindex(list(zip(ch["pitcher_id"], ch["season"]))).values
ch["pitches_prev"] = workload.reindex(list(zip(ch["pitcher_id"], ch["prev_season"]))).values
busy = ch[(ch["pitches_now"] >= 1000) & (ch["pitches_prev"] >= 1000)]
cols = ["player_name", "pitch_type", "usage_pct_prev", "usage_pct", "usage_pct_delta",
        "avg_speed_delta", "is_new"]
busy.nlargest(15, "usage_pct_delta")[cols].reset_index(drop=True)

# %% [markdown]
# ## Summary
#
# The three files support:
# - one pitcher's arsenal over time (`pitcher_arsenal_wide.csv` or `pitcher_arsenal.csv`)
# - league-wide trends in pitch mix, speed and whiff% (`pitcher_arsenal.csv`, weighted by `pitches`)
# - season-to-season changes, including pitches added or dropped (`arsenal_changes.csv`)
# - features for models of pitcher performance
#
# Data: Baseball Savant (MLB Advanced Media). Build script and checks: [kaggle-datasets/pitcher-arsenal-dataset](https://github.com/yasumorishima/kaggle-datasets/tree/main/pitcher-arsenal-dataset).
