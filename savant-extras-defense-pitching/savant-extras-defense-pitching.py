# %% [markdown]
# # Defense & Pitching Quality Analysis with savant-extras
#
# This notebook analyzes defense and pitching quality metrics using
# **[savant-extras](https://github.com/yasumorishima/savant-extras)** v0.4.3 and **[pybaseball](https://github.com/jldbc/pybaseball)**:
#
# - **Outs Above Average (OAA)** — who saves the most outs on defense? (via pybaseball)
# - **Outfield Jump** — first-step reaction and routing efficiency (via pybaseball)
# - **Park Factors** — which parks favor hitters vs pitchers? (via savant-extras)
# - **Pitcher Quality (Stuff+ / Location+ / Pitching+)** — model-based pitcher ratings (via pybaseball)
#
# Data: 2024–2025 seasons via the [Baseball Savant Leaderboards Dataset](https://www.kaggle.com/datasets/yasunorim/baseball-savant-leaderboards-2024).
#
# ```
# pip install savant-extras pybaseball
# ```

# %% [markdown]
# ## Setup

# %%
!pip install -q savant-extras pybaseball

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings
import os
import time

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

YEARS = [2024, 2025]
YEAR = 2025
TOP_N = 20

def coerce_numeric(df):
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > df[col].notna().sum() * 0.5:
                df[col] = converted
    return df

# Dataset path
DATASET_DIR = "/kaggle/input/baseball-savant-leaderboards-2024"

# %% [markdown]
# ---
# ## Part 1: Outs Above Average (OAA)
#
# OAA measures how many outs a fielder saves (positive) or costs (negative) relative to an average fielder,
# based on catch probability of each batted ball.

# %%
oaa_path = os.path.join(DATASET_DIR, "outs_above_average_2024_2025.csv")
if os.path.exists(oaa_path) and os.path.getsize(oaa_path) > 10:
    df_oaa = pd.read_csv(oaa_path)
else:
    try:
        from pybaseball import statcast_outs_above_average
        frames = []
        for y in YEARS:
            df_tmp = statcast_outs_above_average(y, 'all')
            df_tmp = coerce_numeric(df_tmp)
            frames.append(df_tmp)
            time.sleep(1)
        df_oaa = pd.concat(frames, ignore_index=True)
    except Exception as e:
        print(f"outs_above_average: skipped ({e})")
        df_oaa = pd.DataFrame()

print(f"OAA: {len(df_oaa)} fielder-seasons, columns: {list(df_oaa.columns[:8])}")
df_oaa.head()

# %%
# --- Top and Bottom defenders ---
year_col = "year" if "year" in df_oaa.columns else "Year"
oaa25 = df_oaa[df_oaa[year_col] == YEAR].copy()
oaa25 = coerce_numeric(oaa25)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

top_oaa = oaa25.nlargest(TOP_N, "outs_above_average")
bot_oaa = oaa25.nsmallest(TOP_N, "outs_above_average")

name_col = "last_name, first_name" if "last_name, first_name" in oaa25.columns else oaa25.columns[0]

axes[0].barh(top_oaa[name_col], top_oaa["outs_above_average"],
             color=sns.color_palette("crest", TOP_N))
axes[0].set_xlabel("Outs Above Average", fontsize=14)
axes[0].set_title(f"Top {TOP_N} Defenders — {YEAR}", fontsize=16)
axes[0].invert_yaxis()

axes[1].barh(bot_oaa[name_col], bot_oaa["outs_above_average"],
             color=sns.color_palette("flare_r", TOP_N))
axes[1].set_xlabel("Outs Above Average", fontsize=14)
axes[1].set_title(f"Bottom {TOP_N} Defenders — {YEAR}", fontsize=16)
axes[1].invert_yaxis()

plt.suptitle(f"Outs Above Average — {YEAR}", fontsize=18, y=1.01)
plt.tight_layout()
plt.show()

# %%
# --- OAA by position ---
if "primary_pos_formatted" in oaa25.columns:
    pos_order = ["LF", "CF", "RF", "SS", "3B", "2B", "1B"]
    pos_data = oaa25[oaa25["primary_pos_formatted"].isin(pos_order)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=pos_data, x="primary_pos_formatted", y="outs_above_average",
                order=pos_order, ax=axes[0], palette="Set2")
    axes[0].axhline(0, color="gray", lw=1.5, ls="--")
    axes[0].set_xlabel("Position", fontsize=14)
    axes[0].set_ylabel("Outs Above Average", fontsize=14)
    axes[0].set_title(f"OAA Distribution by Position — {YEAR}", fontsize=16)

    if "fielding_runs_prevented" in pos_data.columns:
        for pos in pos_order:
            sub = pos_data[pos_data["primary_pos_formatted"] == pos]
            axes[1].scatter(sub["outs_above_average"], sub["fielding_runs_prevented"],
                            alpha=0.6, s=30, label=pos)
        axes[1].axhline(0, color="gray", lw=0.8, ls="--")
        axes[1].axvline(0, color="gray", lw=0.8, ls="--")
        axes[1].set_xlabel("Outs Above Average", fontsize=14)
        axes[1].set_ylabel("Fielding Runs Prevented", fontsize=14)
        axes[1].set_title(f"OAA vs Fielding Runs Prevented — {YEAR}", fontsize=16)
        axes[1].legend(title="Position", fontsize=10)

    plt.tight_layout()
    plt.show()

# %%
# --- 2024 vs 2025 OAA comparison (players who appear both years) ---
if len(df_oaa[df_oaa[year_col] == 2024]) > 0 and len(oaa25) > 0:
    oaa24 = df_oaa[df_oaa[year_col] == 2024].copy()
    oaa24 = coerce_numeric(oaa24)
    id_col = "player_id" if "player_id" in oaa24.columns else None
    if id_col:
        merged = oaa24[[id_col, name_col, "outs_above_average"]].merge(
            oaa25[[id_col, "outs_above_average"]],
            on=id_col, suffixes=("_2024", "_2025")
        )
        if len(merged) >= 10:
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.scatter(merged["outs_above_average_2024"], merged["outs_above_average_2025"],
                       alpha=0.6, s=30, color="steelblue")
            lim = abs(merged[["outs_above_average_2024", "outs_above_average_2025"]]).max().max() + 3
            ax.plot([-lim, lim], [-lim, lim], "--", color="gray", alpha=0.6)
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            ax.set_xlabel("OAA 2024", fontsize=14)
            ax.set_ylabel("OAA 2025", fontsize=14)
            ax.set_title("OAA Year-over-Year: 2024 vs 2025", fontsize=16)
            plt.tight_layout()
            plt.show()

# %% [markdown]
# ---
# ## Part 2: Outfield Jump
#
# Measures how well outfielders read and react to batted balls in the first 3 seconds after contact.
# Only includes Two-Star plays (≥90% catch probability) — where elite instincts matter most.

# %%
oj_path = os.path.join(DATASET_DIR, "outfield_jump_2024_2025.csv")
if os.path.exists(oj_path) and os.path.getsize(oj_path) > 10:
    df_oj = pd.read_csv(oj_path)
else:
    try:
        from pybaseball import statcast_outfielder_jump
        frames = []
        for y in YEARS:
            df_tmp = statcast_outfielder_jump(y)
            df_tmp = coerce_numeric(df_tmp)
            if "year" not in df_tmp.columns:
                df_tmp["year"] = y
            frames.append(df_tmp)
            time.sleep(1)
        df_oj = pd.concat(frames, ignore_index=True)
    except Exception as e:
        print(f"outfield_jump: skipped ({e})")
        df_oj = pd.DataFrame()

df_oj = coerce_numeric(df_oj)
print(f"Outfield Jump: {len(df_oj)} outfielder-seasons, columns: {list(df_oj.columns[:8])}")
df_oj.head()

# %%
oj_year_col = "year" if "year" in df_oj.columns else "Year"
oj25 = df_oj[df_oj[oj_year_col] == YEAR].copy()
oj_name_col = "player_name" if "player_name" in oj25.columns else df_oj.columns[0]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Top by total jump
if "rel_league_bootup_distance" in oj25.columns:
    top_oj = oj25.nlargest(TOP_N, "rel_league_bootup_distance")
    axes[0].barh(top_oj[oj_name_col], top_oj["rel_league_bootup_distance"],
                 color=sns.color_palette("rocket", TOP_N))
    axes[0].axvline(0, color="gray", lw=1, ls="--")
    axes[0].set_xlabel("Jump vs League Avg (ft)", fontsize=14)
    axes[0].set_title(f"Top {TOP_N} Outfield Jump — {YEAR}", fontsize=16)
    axes[0].invert_yaxis()

# Reaction vs Routing
if "rel_league_reaction_distance" in oj25.columns and "rel_league_routing_distance" in oj25.columns:
    sc = axes[1].scatter(
        oj25["rel_league_reaction_distance"],
        oj25["rel_league_routing_distance"],
        c=oj25.get("rel_league_bootup_distance", pd.Series([0]*len(oj25))),
        cmap="RdYlGn", alpha=0.7, s=50, edgecolors="none"
    )
    plt.colorbar(sc, ax=axes[1], label="Total Jump vs Avg (ft)")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("Reaction Distance vs Avg (ft)", fontsize=14)
    axes[1].set_ylabel("Routing Distance vs Avg (ft)", fontsize=14)
    axes[1].set_title(f"First Step vs Route Efficiency — {YEAR}", fontsize=16)

plt.suptitle(f"Outfield Jump — {YEAR}", fontsize=18, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Breakdown: Reaction / Burst / Routing components
components = {
    "Reaction\n(0–1.5s)": "rel_league_reaction_distance",
    "Burst\n(1.5–3.0s)": "rel_league_burst_distance",
    "Routing": "rel_league_routing_distance",
}
available = {k: v for k, v in components.items() if v in oj25.columns}
if len(available) >= 2:
    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 6))
    if len(available) == 1:
        axes = [axes]
    for ax, (label, col) in zip(axes, available.items()):
        top = oj25.nlargest(15, col)
        ax.barh(top[oj_name_col], top[col], color="steelblue")
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(f"{label} vs Avg (ft)", fontsize=13)
        ax.set_title(f"Top 15: {label} — {YEAR}", fontsize=15)
        ax.invert_yaxis()
    plt.suptitle(f"Jump Components — {YEAR}", fontsize=17)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ## Part 3: Park Factors
#
# FanGraphs park factors — 100 = neutral, >100 = hitter-friendly, <100 = pitcher-friendly.

# %%
pf_path = os.path.join(DATASET_DIR, "park_factors_2024_2025.csv")
if os.path.exists(pf_path) and os.path.getsize(pf_path) > 10:
    df_pf = pd.read_csv(pf_path)
else:
    try:
        from savant_extras import park_factors_range
        df_pf = park_factors_range(2024, 2025)
    except Exception as e:
        print(f"park_factors: skipped ({e})")
        df_pf = pd.DataFrame()

df_pf = coerce_numeric(df_pf)
print(f"Park Factors: {len(df_pf)} team-seasons")
df_pf.head()

# %%
if len(df_pf) == 0 or "season" not in df_pf.columns:
    print("Park Factors: no data available, skipping visualizations")
    pf25 = pd.DataFrame()
else:
    pf25 = df_pf[df_pf["season"] == YEAR].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Full park factor bar — sorted
    pf_sorted = pf25.sort_values("pf_5yr")
    bar_colors = ["#e74c3c" if v > 100 else "#3498db" for v in pf_sorted["pf_5yr"]]
    axes[0].barh(pf_sorted["team"], pf_sorted["pf_5yr"], color=bar_colors)
    axes[0].axvline(100, color="black", lw=1.5, ls="--")
    axes[0].set_xlabel("5-Year Park Factor (runs, 100=neutral)", fontsize=14)
    axes[0].set_title(f"All 30 Teams — Park Factors — {YEAR}", fontsize=16)

    # Multi-factor scatter: HR vs FIP park factor
    if "pf_hr" in pf25.columns and "pf_fip" in pf25.columns:
        scatter_colors = ["#e74c3c" if v > 100 else "#3498db" for v in pf25["pf_5yr"]]
        axes[1].scatter(pf25["pf_hr"], pf25["pf_fip"], c=scatter_colors, s=80, alpha=0.8, edgecolors="white", lw=0.5)
        for _, row in pf25.iterrows():
            axes[1].annotate(row["team"], (row["pf_hr"], row["pf_fip"]),
                             fontsize=7.5, ha="center", va="bottom")
        axes[1].axhline(100, color="gray", lw=0.8, ls="--")
        axes[1].axvline(100, color="gray", lw=0.8, ls="--")
        axes[1].set_xlabel("HR Park Factor", fontsize=14)
        axes[1].set_ylabel("FIP Park Factor", fontsize=14)
        axes[1].set_title(f"HR vs FIP Park Factor — {YEAR}", fontsize=16)

    plt.tight_layout()
    plt.show()

# %%
# Year-over-year change in park factors
if len(df_pf) > 0 and "season" in df_pf.columns and len(df_pf[df_pf["season"] == 2024]) > 0:
    pf24 = df_pf[df_pf["season"] == 2024][["team", "pf_5yr", "pf_hr"]].rename(
        columns={"pf_5yr": "pf_5yr_2024", "pf_hr": "pf_hr_2024"})
    pf25_slim = pf25[["team", "pf_5yr", "pf_hr"]].rename(
        columns={"pf_5yr": "pf_5yr_2025", "pf_hr": "pf_hr_2025"})
    pf_comp = pf24.merge(pf25_slim, on="team")
    pf_comp["delta_5yr"] = pf_comp["pf_5yr_2025"] - pf_comp["pf_5yr_2024"]
    pf_comp = pf_comp.sort_values("delta_5yr")

    fig, ax = plt.subplots(figsize=(10, 8))
    bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in pf_comp["delta_5yr"]]
    ax.barh(pf_comp["team"], pf_comp["delta_5yr"], color=bar_colors)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Change in 5-Year Park Factor (2024 → 2025)", fontsize=14)
    ax.set_title("Park Factor Change: 2024 vs 2025", fontsize=16)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ## Part 4: Pitcher Quality — Stuff+ / Location+ / Pitching+
#
# FanGraphs model-based metrics (available from 2020+):
# - **Stuff+**: physical pitch quality (velocity, movement, spin)
# - **Location+**: pitch location within counts and pitch types
# - **Pitching+**: combined overall quality
#
# 100 = MLB average. Available via pybaseball's fg_pitching_data().

# %%
pq_path = os.path.join(DATASET_DIR, "pitcher_quality_2024_2025.csv")
if os.path.exists(pq_path) and os.path.getsize(pq_path) > 10:
    df_pq = pd.read_csv(pq_path)
else:
    try:
        from pybaseball import fg_pitching_data
        _PQ_COLS = ["Name", "Team", "Age", "IP", "Stuff+", "Location+", "Pitching+"]
        _PQ_RENAME = {"Name": "name", "Team": "team", "Age": "age", "IP": "ip",
                      "Stuff+": "stuff_plus", "Location+": "location_plus", "Pitching+": "pitching_plus"}
        frames = []
        for y in YEARS:
            df_tmp = fg_pitching_data(y, qual=0)
            avail = [c for c in _PQ_COLS if c in df_tmp.columns]
            df_tmp = df_tmp[avail].rename(columns=_PQ_RENAME).copy()
            df_tmp = coerce_numeric(df_tmp)
            df_tmp["season"] = y
            frames.append(df_tmp)
            time.sleep(1.5)
        df_pq = pd.concat(frames, ignore_index=True)
    except Exception as e:
        print(f"pitcher_quality: skipped ({e})")
        df_pq = pd.DataFrame()

df_pq = coerce_numeric(df_pq)
print(f"Pitcher Quality: {len(df_pq)} pitcher-seasons")
df_pq.head()

# %%
pq_season_col = "season" if "season" in df_pq.columns else "Season"
pq25 = df_pq[df_pq[pq_season_col] == YEAR].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Top Pitching+
if "pitching_plus" in pq25.columns:
    top_pq = pq25.nlargest(TOP_N, "pitching_plus")
    axes[0].barh(top_pq["name"], top_pq["pitching_plus"],
                 color=sns.color_palette("flare", TOP_N))
    axes[0].axvline(100, color="gray", lw=1.5, ls="--")
    axes[0].set_xlabel("Pitching+ (100 = MLB avg)", fontsize=14)
    axes[0].set_title(f"Top {TOP_N} Pitching+ — {YEAR}", fontsize=16)
    axes[0].invert_yaxis()

# Stuff+ vs Location+
if "stuff_plus" in pq25.columns and "location_plus" in pq25.columns:
    sc_color = pq25.get("pitching_plus", pd.Series([100]*len(pq25)))
    sc = axes[1].scatter(pq25["stuff_plus"], pq25["location_plus"],
                         c=sc_color, cmap="RdYlGn", vmin=70, vmax=130,
                         alpha=0.6, s=30, edgecolors="none")
    plt.colorbar(sc, ax=axes[1], label="Pitching+")
    axes[1].axhline(100, color="gray", lw=0.8, ls="--")
    axes[1].axvline(100, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("Stuff+ (100 = avg)", fontsize=14)
    axes[1].set_ylabel("Location+ (100 = avg)", fontsize=14)
    axes[1].set_title(f"Stuff+ vs Location+ — {YEAR}", fontsize=16)

plt.suptitle(f"Pitcher Quality (Stuff+ / Location+ / Pitching+) — {YEAR}", fontsize=18, y=1.01)
plt.tight_layout()
plt.show()

# %%
# Per-pitch Stuff+ comparison (SP starters with high IP)
stuff_cols = [c for c in pq25.columns if c.startswith("stuff_") and c != "stuff_plus"]
if len(stuff_cols) >= 3:
    min_ip = 100
    starters = pq25[pq25.get("ip", pd.Series([0]*len(pq25))) >= min_ip] if "ip" in pq25.columns else pq25
    top_starters = starters.nlargest(15, "stuff_plus") if "stuff_plus" in starters.columns else starters.head(15)

    pitch_labels = {
        "stuff_fa": "4-Seam FB", "stuff_si": "Sinker", "stuff_fc": "Cutter",
        "stuff_sl": "Slider", "stuff_cu": "Curveball", "stuff_ch": "Changeup",
        "stuff_st": "Sweeper", "stuff_fs": "Splitter",
    }
    available_cols = [c for c in stuff_cols if c in pitch_labels]
    if available_cols:
        heat_data = top_starters.set_index("name")[available_cols].rename(columns=pitch_labels)
        heat_data = heat_data.apply(pd.to_numeric, errors="coerce")

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(heat_data, annot=True, fmt=".0f", cmap="RdYlGn",
                    center=100, vmin=70, vmax=130,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Stuff+"})
        ax.set_title(f"Per-Pitch Stuff+ — Top 15 Starters — {YEAR}", fontsize=16)
        ax.set_xlabel("Pitch Type", fontsize=13)
        ax.set_ylabel("")
        plt.tight_layout()
        plt.show()

# %%
# Year-over-year Pitching+ change
if len(df_pq[df_pq[pq_season_col] == 2024]) > 0 and "mlbam_id" in df_pq.columns:
    pq24 = df_pq[df_pq[pq_season_col] == 2024][["mlbam_id", "name", "pitching_plus"]].rename(
        columns={"pitching_plus": "pitching_plus_2024"})
    pq25_slim = pq25[["mlbam_id", "pitching_plus"]].rename(
        columns={"pitching_plus": "pitching_plus_2025"})
    pq_comp = pq24.merge(pq25_slim, on="mlbam_id").dropna()
    pq_comp["delta"] = pq_comp["pitching_plus_2025"] - pq_comp["pitching_plus_2024"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(pq_comp["pitching_plus_2024"], pq_comp["pitching_plus_2025"],
                    alpha=0.5, s=20, color="purple")
    lims = [pq_comp[["pitching_plus_2024", "pitching_plus_2025"]].min().min() - 5,
            pq_comp[["pitching_plus_2024", "pitching_plus_2025"]].max().max() + 5]
    axes[0].plot(lims, lims, "--", color="gray", alpha=0.6)
    axes[0].axhline(100, color="gray", lw=0.5, ls=":")
    axes[0].axvline(100, color="gray", lw=0.5, ls=":")
    axes[0].set_xlabel("Pitching+ 2024", fontsize=14)
    axes[0].set_ylabel("Pitching+ 2025", fontsize=14)
    axes[0].set_title("Pitching+ Year-over-Year", fontsize=16)

    top_rise = pq_comp.nlargest(15, "delta")
    axes[1].barh(top_rise["name"], top_rise["delta"],
                 color=sns.color_palette("crest", 15))
    axes[1].axvline(0, color="gray", lw=1, ls="--")
    axes[1].set_xlabel("Pitching+ Change (2024 → 2025)", fontsize=14)
    axes[1].set_title("Biggest Improvers — Pitching+ 2024→2025", fontsize=16)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ## Summary
#
# | Leaderboard | Source | Key Metric |
# |---|---|---|
# | Outs Above Average | Baseball Savant | `outs_above_average` (OAA) |
# | Outfield Jump | Baseball Savant | `rel_league_bootup_distance` (ft vs avg) |
# | Park Factors | FanGraphs Guts! | `pf_5yr` (100 = neutral) |
# | Pitcher Quality | FanGraphs | `stuff_plus`, `location_plus`, `pitching_plus` |
#
# All data fetched via **savant-extras** — install with `pip install savant-extras`.
#
# - **PyPI**: [savant-extras](https://pypi.org/project/savant-extras/)
# - **GitHub**: [yasumorishima/savant-extras](https://github.com/yasumorishima/savant-extras)
# - **Dataset**: [Baseball Savant Leaderboards 2024-2025](https://www.kaggle.com/datasets/yasunorim/baseball-savant-leaderboards-2024)
