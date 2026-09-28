# %% [markdown]
# # Defense & Pitching Quality from Baseball Savant Leaderboards
#
# This notebook looks at four leaderboards from the dataset **[Baseball Savant Leaderboards (2024-2026)](https://www.kaggle.com/datasets/yasunorim/baseball-savant-leaderboards-2024)**, built with **[savant-extras](https://github.com/yasumorishima/savant-extras)** 0.6.0 and Savant's CSV endpoints:
#
# | File | Seasons | Source |
# |---|---|---|
# | `outs_above_average.csv` | 2024-2026 | Baseball Savant, Outs Above Average |
# | `outfield_jump.csv` | 2024-2026 | Baseball Savant, Outfielder Jump |
# | `park_factors.csv` | 2024-2026 | Baseball Savant, Statcast Park Factors (`savant_extras.park_factors`) |
# | `pitcher_quality.csv` | 2024-2025 | FanGraphs Stuff+ / Location+ / Pitching+ |
#
# The dataset was rebuilt in September 2026. File names no longer carry the seasons (`outs_above_average.csv`, not `outs_above_average_2024_2025.csv`); every file has a `year` or `season` column. 2026 is the complete regular season (it ended on September 27).
#
# Two changes matter for this notebook:
# - **Park factors now come from Baseball Savant, not FanGraphs.** The columns `pf_5yr` and `pf_fip` no longer exist; Savant gives a 1-year and a 3-year runs factor (`pf_1yr`, `pf_3yr`) and component factors on the 3-year window.
# - **Pitcher quality stops at 2025.** FanGraphs refuses the build machines, so that table was carried over unchanged.

# %% [markdown]
# ## Setup

# %%
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
                     "legend.title_fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12})
warnings.filterwarnings("ignore", category=FutureWarning)

YEARS = [2024, 2025, 2026]
YEAR = 2026        # single-season charts
PQ_YEAR = 2025     # pitcher_quality ends in 2025
TOP_N = 20

# On Kaggle the dataset is under /kaggle/input/; KAGGLE_INPUT_ROOT lets the notebook run elsewhere.
INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
paths = glob.glob(os.path.join(INPUT_ROOT, "**", "outs_above_average.csv"), recursive=True)
if not paths:
    raise FileNotFoundError("Attach the dataset yasunorim/baseball-savant-leaderboards-2024 to this notebook.")
DATASET_DIR = os.path.dirname(paths[0])


def load(name):
    df = pd.read_csv(os.path.join(DATASET_DIR, f"{name}.csv"))
    season_col = "year" if "year" in df.columns else "season"
    print(f"{name}: {len(df):,} rows, seasons {sorted(df[season_col].unique().tolist())}")
    return df


def barh(ax, df, label, value, title, xlabel, palette="crest", n=TOP_N, smallest=False):
    top = (df.nsmallest if smallest else df.nlargest)(n, value)
    ax.barh(top[label], top[value], color=sns.color_palette(palette, len(top)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

# %% [markdown]
# ---
# ## Part 1: Outs Above Average (OAA)
#
# OAA measures how many outs a fielder saves (positive) or costs (negative) relative to an average fielder, based on the catch probability of each batted ball. Qualified fielders only.

# %%
df_oaa = load("outs_above_average")
NAME = "last_name, first_name"
df_oaa.head()

# %%
oaa = df_oaa[df_oaa["year"] == YEAR]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
barh(axes[0], oaa, NAME, "outs_above_average", f"Top {TOP_N} defenders, {YEAR}", "Outs Above Average")
barh(axes[1], oaa, NAME, "outs_above_average", f"Bottom {TOP_N} defenders, {YEAR}", "Outs Above Average",
     palette="flare_r", smallest=True)
plt.tight_layout()
plt.show()

# %%
pos_order = ["LF", "CF", "RF", "SS", "3B", "2B", "1B"]
pos_data = oaa[oaa["primary_pos_formatted"].isin(pos_order)]
pos_colors = dict(zip(pos_order, sns.color_palette("Set2", len(pos_order))))

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
sns.boxplot(data=pos_data, x="primary_pos_formatted", y="outs_above_average", order=pos_order,
            hue="primary_pos_formatted", palette=pos_colors, dodge=False, ax=axes[0])
if axes[0].get_legend() is not None:
    axes[0].get_legend().remove()
axes[0].axhline(0, color="gray", lw=1.5, ls="--")
axes[0].set_xlabel("Position")
axes[0].set_ylabel("Outs Above Average")
axes[0].set_title(f"OAA by position, {YEAR}")

for pos in pos_order:
    sub = pos_data[pos_data["primary_pos_formatted"] == pos]
    axes[1].scatter(sub["outs_above_average"], sub["fielding_runs_prevented"], alpha=0.7, s=35,
                    color=pos_colors[pos], label=pos)
axes[1].axhline(0, color="gray", lw=0.8, ls="--")
axes[1].axvline(0, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("Outs Above Average")
axes[1].set_ylabel("Fielding Runs Prevented")
axes[1].set_title(f"OAA vs Fielding Runs Prevented, {YEAR}")
axes[1].legend(title="Position", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %%
# Year over year: qualified fielders in both seasons (one row per player and season; the sum guards against a split row)
per_year = df_oaa.groupby(["player_id", "year"])["outs_above_average"].sum().unstack()
pairs = [(2024, 2025), (2025, 2026)]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, (a, b) in zip(axes, pairs):
    m = per_year[[a, b]].dropna()
    r = m[a].corr(m[b])
    ax.scatter(m[a], m[b], alpha=0.6, s=30, color="steelblue")
    lim = m.abs().max().max() + 3
    ax.plot([-lim, lim], [-lim, lim], "--", color="gray", alpha=0.6)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"OAA {a}")
    ax.set_ylabel(f"OAA {b}")
    ax.set_title(f"OAA {a} vs {b} (n = {len(m)}, r = {r:.2f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Part 2: Outfield Jump
#
# Jump measures how many feet an outfielder covers in the right direction in the first 3 seconds after the pitch is released, relative to league average. Savant computes it only on plays with a catch probability of 90% or lower ("two stars" or harder). It splits into:
# - **Reaction**: feet covered in the first 1.5 seconds
# - **Burst**: feet covered in the next 1.5 seconds
# - **Route**: feet covered in the right direction compared with feet covered in any direction, over the full 3 seconds
#
# The table comes from Savant's CSV, qualified outfielders only.

# %%
df_oj = load("outfield_jump")
df_oj.head()

# %%
oj = df_oj[df_oj["year"] == YEAR]

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
barh(axes[0], oj, NAME, "rel_league_bootup_distance", f"Top {TOP_N} outfield jump, {YEAR}",
     "Jump vs league average (ft)", palette="rocket")
axes[0].axvline(0, color="gray", lw=1, ls="--")

sc = axes[1].scatter(oj["rel_league_reaction_distance"], oj["rel_league_routing_distance"],
                     c=oj["rel_league_bootup_distance"], cmap="RdYlGn", vmin=-4, vmax=4,
                     alpha=0.8, s=55, edgecolors="none")
plt.colorbar(sc, ax=axes[1], label="Jump vs average (ft)")
axes[1].axhline(0, color="gray", lw=0.8, ls="--")
axes[1].axvline(0, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("Reaction vs average (ft)")
axes[1].set_ylabel("Route vs average (ft)")
axes[1].set_title(f"Reaction vs route, {YEAR} (n = {len(oj)})")
plt.tight_layout()
plt.show()

# %%
components = {
    "Reaction (0-1.5 s)": "rel_league_reaction_distance",
    "Burst (1.5-3 s)": "rel_league_burst_distance",
    "Route (0-3 s)": "rel_league_routing_distance",
}
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, (label, col) in zip(axes, components.items()):
    barh(ax, oj, NAME, col, f"Top 15: {label}", f"{label.split(' (')[0]} vs average (ft)",
         palette="Blues_r", n=15)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
plt.suptitle(f"Jump components, {YEAR}", fontsize=18)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Part 3: Park Factors
#
# Baseball Savant's Statcast park factors: 100 = neutral, above 100 = hitter-friendly, below 100 = pitcher-friendly.
# - `pf_1yr`: runs factor for the season alone
# - `pf_3yr`: runs factor over the 3 seasons ending with that season (`pf_3yr_years`); the component factors (`pf_hr`, `pf_so`, ...) also use this window
#
# A park without three seasons of history has no 3-year columns: OAK 2025 and 2026 (Sutter Health Park) and TB 2025 (Steinbrenner Field). Their `pf_1yr` is filled.

# %%
df_pf = load("park_factors")
print("\nNo 3-year window:")
print(df_pf[df_pf["pf_3yr"].isna()][["team", "year", "venue_name", "pf_1yr"]].to_string(index=False))
df_pf.head()

# %%
pf = df_pf[df_pf["year"] == YEAR].sort_values("pf_1yr").reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(17, 9))
y = np.arange(len(pf))
axes[0].hlines(y, pf[["pf_1yr", "pf_3yr"]].min(axis=1), pf[["pf_1yr", "pf_3yr"]].max(axis=1), color="lightgray", lw=2)
axes[0].scatter(pf["pf_1yr"], y, color="#e74c3c", s=60, label=f"{YEAR} alone", zorder=3)
axes[0].scatter(pf["pf_3yr"], y, color="#3498db", s=60, label="3 years", zorder=3)
axes[0].set_yticks(y)
axes[0].set_yticklabels(pf["team"])
axes[0].axvline(100, color="black", lw=1.2, ls="--")
axes[0].set_xlabel("Runs park factor (100 = neutral)")
axes[0].set_title(f"Runs park factor, {YEAR}")
axes[0].legend(loc="lower right")

pf3 = pf.dropna(subset=["pf_hr", "pf_so"])
sc = axes[1].scatter(pf3["pf_hr"], pf3["pf_so"], c=pf3["pf_3yr"], cmap="coolwarm", vmin=85, vmax=115,
                     s=90, edgecolors="gray", lw=0.5)
plt.colorbar(sc, ax=axes[1], label="3-year runs factor")
# Parks that share a point get one joined label; labels of close neighbours are pushed apart
pts = pf3.groupby(["pf_hr", "pf_so"])["team"].agg("/".join).reset_index()
for _, row in pts.iterrows():
    near = pts[(abs(pts["pf_hr"] - row["pf_hr"]) < 3) & (abs(pts["pf_so"] - row["pf_so"]) < 1) &
               (pts["pf_hr"] != row["pf_hr"])]
    if near.empty:
        ha, dx = "center", 0
    elif (near["pf_hr"] > row["pf_hr"]).all():
        ha, dx = "right", -2
    else:
        ha, dx = "left", 2
    axes[1].annotate(row["team"], (row["pf_hr"], row["pf_so"]), fontsize=10, ha=ha, va="bottom",
                     xytext=(dx, 5), textcoords="offset points")
axes[1].axhline(100, color="gray", lw=0.8, ls="--")
axes[1].axvline(100, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("HR park factor (3 years)")
axes[1].set_ylabel("Strikeout park factor (3 years)")
axes[1].set_title(f"HR vs strikeout factor, {pf3['pf_3yr_years'].iloc[0]}")
plt.tight_layout()
plt.show()
print("Not in the right panel (no 3-year window):",
      ", ".join(pf.loc[pf["pf_hr"].isna(), "team"]))

# %%
# Change in the 3-year runs factor from 2025 to 2026 (parks with a 3-year window in both seasons)
w = df_pf.pivot(index="team", columns="year", values="pf_3yr")
comp = (w[2026] - w[2025]).dropna().sort_values()
missing = sorted(set(w.index) - set(comp.index))

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(comp.index, comp.values, color=["#e74c3c" if v > 0 else "#3498db" for v in comp.values])
ax.axvline(0, color="black", lw=1.2)
ax.set_xlabel("Change in 3-year runs park factor")
ax.set_title("3-year park factor: 2025 to 2026")
plt.tight_layout()
plt.show()
print("Not shown (no 3-year window in 2025 or 2026):", ", ".join(missing))

# %% [markdown]
# ---
# ## Part 4: Pitcher Quality: Stuff+ / Location+ / Pitching+
#
# FanGraphs model-based metrics, 100 = MLB average:
# - **Stuff+**: physical pitch quality (velocity, movement, spin)
# - **Location+**: where pitches are thrown, given count and pitch type
# - **Pitching+**: both combined
#
# The table covers **2024-2025 only**. It includes every pitcher, down to a few innings, and those small samples give extreme values (Pitching+ ranges from -30 to 158). The charts below use pitchers with at least 50 innings.

# %%
df_pq = load("pitcher_quality")
MIN_IP = 50
df_pq.head()

# %%
pq = df_pq[df_pq["season"] == PQ_YEAR]
pq_q = pq[pq["ip"] >= MIN_IP]
print(f"{PQ_YEAR}: {len(pq)} pitchers, {len(pq_q)} with {MIN_IP}+ IP")
print("\nHighest Pitching+ without an innings filter:")
print(pq.nlargest(5, "pitching_plus")[["name", "team", "ip", "pitching_plus"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(17, 8))
# Bars start at 100 (league average), so their length is the distance above average
top = pq_q.nlargest(TOP_N, "pitching_plus")
axes[0].barh(top["name"], top["pitching_plus"] - 100, left=100, color=sns.color_palette("flare", len(top)))
axes[0].invert_yaxis()
axes[0].set_xlim(100, top["pitching_plus"].max() + 3)
axes[0].set_xlabel("Pitching+ (100 = MLB average)")
axes[0].set_title(f"Top {TOP_N} Pitching+, {PQ_YEAR} ({MIN_IP}+ IP)")

sc = axes[1].scatter(pq_q["stuff_plus"], pq_q["location_plus"], c=pq_q["pitching_plus"],
                     cmap="RdYlGn", vmin=85, vmax=115, alpha=0.75, s=35, edgecolors="none")
plt.colorbar(sc, ax=axes[1], label="Pitching+")
axes[1].axhline(100, color="gray", lw=0.8, ls="--")
axes[1].axvline(100, color="gray", lw=0.8, ls="--")
axes[1].set_xlabel("Stuff+ (100 = average)")
axes[1].set_ylabel("Location+ (100 = average)")
axes[1].set_title(f"Stuff+ vs Location+, {PQ_YEAR} ({MIN_IP}+ IP, n = {len(pq_q)})")
plt.tight_layout()
plt.show()

# %%
# Year over year. The table has no player id, so pitchers are matched by name;
# names that appear twice among the 50+ IP pitchers of a season are left out.
def unique_names(d):
    return d[~d["name"].duplicated(keep=False)]

a = unique_names(df_pq[(df_pq["season"] == 2024) & (df_pq["ip"] >= MIN_IP)])
b = unique_names(df_pq[(df_pq["season"] == 2025) & (df_pq["ip"] >= MIN_IP)])
comp = a[["name", "pitching_plus"]].merge(b[["name", "pitching_plus"]], on="name", suffixes=("_2024", "_2025"))
comp["delta"] = comp["pitching_plus_2025"] - comp["pitching_plus_2024"]
r = comp["pitching_plus_2024"].corr(comp["pitching_plus_2025"])

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
axes[0].scatter(comp["pitching_plus_2024"], comp["pitching_plus_2025"], alpha=0.6, s=25, color="purple")
lims = [comp[["pitching_plus_2024", "pitching_plus_2025"]].min().min() - 3,
        comp[["pitching_plus_2024", "pitching_plus_2025"]].max().max() + 3]
axes[0].plot(lims, lims, "--", color="gray", alpha=0.6)
axes[0].set_xlim(lims)
axes[0].set_ylim(lims)
axes[0].set_xlabel("Pitching+ 2024")
axes[0].set_ylabel("Pitching+ 2025")
axes[0].set_title(f"Pitching+ 2024 vs 2025 ({MIN_IP}+ IP both, n = {len(comp)}, r = {r:.2f})")

barh(axes[1], comp, "name", "delta", "Largest Pitching+ gains, 2024 to 2025", "Change in Pitching+", n=15)
axes[1].axvline(0, color="gray", lw=1, ls="--")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Summary
#
# | Leaderboard | Source | Seasons | Key metric |
# |---|---|---|---|
# | Outs Above Average | Baseball Savant | 2024-2026 | `outs_above_average` |
# | Outfield Jump | Baseball Savant | 2024-2026 | `rel_league_bootup_distance` (ft vs average) |
# | Park Factors | Baseball Savant | 2024-2026 | `pf_1yr`, `pf_3yr` (100 = neutral) |
# | Pitcher Quality | FanGraphs | 2024-2025 | `stuff_plus`, `location_plus`, `pitching_plus` |
#
# - **Dataset**: [Baseball Savant Leaderboards (2024-2026)](https://www.kaggle.com/datasets/yasunorim/baseball-savant-leaderboards-2024)
# - **PyPI**: [savant-extras](https://pypi.org/project/savant-extras/)
# - **GitHub**: [yasumorishima/savant-extras](https://github.com/yasumorishima/savant-extras)
#
# Data: Baseball Savant (MLB Advanced Media); Stuff+ / Location+ / Pitching+ from FanGraphs.
