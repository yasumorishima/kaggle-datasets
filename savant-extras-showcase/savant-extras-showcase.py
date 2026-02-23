# %% [markdown]
# # savant-extras: All Baseball Savant Leaderboards in One Package
#
# **[savant-extras](https://github.com/yasumorishima/savant-extras)** provides 15+ Baseball Savant leaderboards that [pybaseball](https://github.com/jldbc/pybaseball) doesn't support — all as simple one-line function calls returning DataFrames.
#
# This notebook demonstrates every leaderboard with visualizations using **2024 season data**.
#
# ```
# pip install savant-extras
# ```

# %% [markdown]
# ## Setup

# %%
!pip install -q savant-extras

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import time

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore")

YEAR = 2024
TOP_N = 20  # top N players for bar charts

def coerce_numeric(df):
    """Convert columns that look numeric but are stored as object dtype."""
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > df[col].notna().sum() * 0.5:
                df[col] = converted
    return df

# %% [markdown]
# ---
# ## 1. Bat Tracking (2024+)
# Bat speed, attack angle, swing tilt — with custom date ranges.

# %%
from savant_extras import bat_tracking

df_bat = bat_tracking("2024-04-01", "2024-09-30", min_swings=100)
df_bat = coerce_numeric(df_bat)
print(f"Bat Tracking: {len(df_bat)} players")
df_bat.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top = df_bat.nlargest(TOP_N, "avg_bat_speed")
ax.barh(top["name"], top["avg_bat_speed"], color=sns.color_palette("rocket", TOP_N))
ax.set_xlabel("Average Bat Speed (mph)")
ax.set_title(f"Top {TOP_N} Bat Speed — {YEAR}")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_bat["avg_bat_speed"], df_bat["attack_angle"], alpha=0.5, s=20)
ax.set_xlabel("Average Bat Speed (mph)")
ax.set_ylabel("Attack Angle (°)")
ax.set_title(f"Bat Speed vs Attack Angle — {YEAR}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 2. Pitch Tempo (2010+)
# Pace metrics — median seconds between pitches, hot/warm/cold frequency.

# %%
from savant_extras import pitch_tempo

df_tempo = pitch_tempo(YEAR)
df_tempo = coerce_numeric(df_tempo)
print(f"Pitch Tempo: {len(df_tempo)} pitchers")
df_tempo.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fastest pitchers (bases empty)
fastest = df_tempo.nsmallest(TOP_N, "median_seconds_empty")
axes[0].barh(fastest["entity_name"], fastest["median_seconds_empty"],
             color=sns.color_palette("YlOrRd_r", TOP_N))
axes[0].set_xlabel("Median Seconds (Bases Empty)")
axes[0].set_title(f"Fastest Tempo — {YEAR}")
axes[0].invert_yaxis()

# Distribution
axes[1].hist(df_tempo["median_seconds_empty"].dropna(), bins=30, color="steelblue", edgecolor="white")
axes[1].set_xlabel("Median Seconds (Bases Empty)")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Pitch Tempo Distribution — {YEAR}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 3. Arm Strength (2020+)
# Fielder throw speed by position.

# %%
from savant_extras import arm_strength

df_arm = arm_strength(YEAR, min_throws=50)
df_arm = coerce_numeric(df_arm)
print(f"Arm Strength: {len(df_arm)} fielders")
df_arm.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top arm strength
top_arm = df_arm.nlargest(TOP_N, "max_arm_strength")
axes[0].barh(top_arm["fielder_name"], top_arm["max_arm_strength"],
             color=sns.color_palette("flare", TOP_N))
axes[0].set_xlabel("Max Arm Strength (mph)")
axes[0].set_title(f"Top {TOP_N} Max Throw Speed — {YEAR}")
axes[0].invert_yaxis()

# By position
pos_order = ["RF", "CF", "LF", "SS", "3B", "2B", "1B"]
pos_data = df_arm[df_arm["primary_position"].isin(pos_order)]
sns.boxplot(data=pos_data, x="primary_position", y="arm_overall",
            order=pos_order, ax=axes[1], palette="Set2")
axes[1].set_xlabel("Position")
axes[1].set_ylabel("Arm Strength Overall (mph)")
axes[1].set_title(f"Arm Strength by Position — {YEAR}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 4. Batted Ball Profile
# Ground ball, fly ball, line drive rates and pull/oppo splits.

# %%
from savant_extras import batted_ball

df_bb = batted_ball(YEAR)
df_bb = coerce_numeric(df_bb)
print(f"Batted Ball: {len(df_bb)} batters")
df_bb.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# GB vs FB rate
axes[0].scatter(df_bb["gb_rate"], df_bb["fb_rate"], alpha=0.5, s=20, color="teal")
axes[0].set_xlabel("Ground Ball Rate")
axes[0].set_ylabel("Fly Ball Rate")
axes[0].set_title(f"GB Rate vs FB Rate — {YEAR}")

# Pull rate distribution
axes[1].hist(df_bb["pull_rate"].dropna(), bins=25, color="coral", edgecolor="white")
axes[1].set_xlabel("Pull Rate")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Pull Rate Distribution — {YEAR}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 5. Home Runs
# HR distance, exit velocity, expected HR, no-doubters.

# %%
from savant_extras import home_runs

df_hr = home_runs(YEAR)
df_hr = coerce_numeric(df_hr)
print(f"Home Runs: {len(df_hr)} batters")
df_hr.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Most no-doubters
top_nd = df_hr.nlargest(TOP_N, "no_doubters")
axes[0].barh(top_nd["player"], top_nd["no_doubters"],
             color=sns.color_palette("magma", TOP_N))
axes[0].set_xlabel("No-Doubter Home Runs")
axes[0].set_title(f"Top {TOP_N} No-Doubters — {YEAR}")
axes[0].invert_yaxis()

# HR vs xHR
axes[1].scatter(df_hr["xhr"], df_hr["hr_total"], alpha=0.5, s=20, color="purple")
lims = [0, df_hr[["xhr", "hr_total"]].max().max() + 5]
axes[1].plot(lims, lims, "--", color="gray", alpha=0.5)
axes[1].set_xlabel("Expected HR (xHR)")
axes[1].set_ylabel("Actual HR")
axes[1].set_title(f"HR vs xHR — {YEAR}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 6. Pitch Movement
# Horizontal and vertical break by pitch type.

# %%
from savant_extras import pitch_movement

df_pm = pitch_movement(YEAR)
df_pm = coerce_numeric(df_pm)
print(f"Pitch Movement: {len(df_pm)} pitcher-pitch combos")
df_pm.head()

# %%
fig, ax = plt.subplots(figsize=(10, 8))
pitch_types = ["FF", "SL", "CU", "CH", "SI", "FC", "ST", "SV"]
colors = sns.color_palette("Set1", len(pitch_types))

for pt, color in zip(pitch_types, colors):
    sub = df_pm[df_pm["pitch_type"] == pt]
    if len(sub) > 0:
        ax.scatter(sub["pitcher_break_x"], sub["pitcher_break_z"],
                   alpha=0.3, s=10, color=color, label=pt)

ax.set_xlabel("Horizontal Break (in)")
ax.set_ylabel("Vertical Break (in)")
ax.set_title(f"Pitch Movement by Type — {YEAR}")
ax.legend(title="Pitch Type", markerscale=3)
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 7. Swing & Take Run Value
# Run values by zone: heart, shadow, chase, waste.

# %%
from savant_extras import swing_take

df_st = swing_take(YEAR)
df_st = coerce_numeric(df_st)
print(f"Swing & Take: {len(df_st)} batters")
df_st.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
df_st["runs_all"] = pd.to_numeric(df_st["runs_all"], errors="coerce")
top_st = df_st.nlargest(TOP_N, "runs_all")
ax.barh(top_st["last_name, first_name"], top_st["runs_all"],
        color=sns.color_palette("viridis", TOP_N))
ax.set_xlabel("Total Run Value (Swing + Take)")
ax.set_title(f"Top {TOP_N} Swing & Take Run Value — {YEAR}")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 6))
zones = ["runs_heart", "runs_shadow", "runs_chase", "runs_waste"]
for z in zones:
    df_st[z] = pd.to_numeric(df_st[z], errors="coerce")
zone_means = [df_st[z].mean() for z in zones]
zone_labels = ["Heart", "Shadow", "Chase", "Waste"]
colors = ["#e74c3c", "#f39c12", "#3498db", "#95a5a6"]
ax.bar(zone_labels, zone_means, color=colors, edgecolor="white", width=0.6)
ax.set_ylabel("Mean Run Value")
ax.set_title(f"Average Run Value by Zone — {YEAR}")
ax.axhline(0, color="black", lw=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 8. Pitcher Arm Angle
# Release point angles and positions.

# %%
from savant_extras import pitcher_arm_angle

df_angle = pitcher_arm_angle(YEAR)
df_angle = coerce_numeric(df_angle)
print(f"Pitcher Arm Angle: {len(df_angle)} pitchers")
df_angle.head()

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(df_angle["ball_angle"].dropna(), bins=30, color="dodgerblue", edgecolor="white")
ax.set_xlabel("Ball Angle (°)")
ax.set_ylabel("Count")
ax.set_title(f"Pitcher Arm Angle Distribution — {YEAR}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 9. Running Game (Pitcher)
# Pitcher's ability to control the running game.

# %%
from savant_extras import running_game

df_rg = running_game(YEAR)
df_rg = coerce_numeric(df_rg)
print(f"Running Game: {len(df_rg)} pitchers")
df_rg.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top_rg = df_rg.nlargest(TOP_N, "runs_prevented_on_running_attr")
ax.barh(top_rg["player_name"], top_rg["runs_prevented_on_running_attr"],
        color=sns.color_palette("crest", TOP_N))
ax.set_xlabel("Runs Prevented on Running")
ax.set_title(f"Top {TOP_N} Pitchers — Running Game — {YEAR}")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 10. Catcher Blocking
# Blocks above average, PB/WP prevention.

# %%
from savant_extras import catcher_blocking

df_cb = catcher_blocking(YEAR)
df_cb = coerce_numeric(df_cb)
print(f"Catcher Blocking: {len(df_cb)} catchers")
df_cb.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top_cb = df_cb.nlargest(15, "blocks_above_average")
ax.barh(top_cb["player_name"], top_cb["blocks_above_average"],
        color=sns.color_palette("mako", 15))
ax.set_xlabel("Blocks Above Average")
ax.set_title(f"Top 15 Catchers — Blocking — {YEAR}")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 11. Catcher Throwing
# Pop time, exchange time, CS rate, arm strength.

# %%
from savant_extras import catcher_throwing

df_ct = catcher_throwing(YEAR)
df_ct = coerce_numeric(df_ct)
print(f"Catcher Throwing: {len(df_ct)} catchers")
df_ct.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pop time (lower is better)
top_pop = df_ct.nsmallest(15, "pop_time")
axes[0].barh(top_pop["player_name"], top_pop["pop_time"],
             color=sns.color_palette("YlOrRd_r", 15))
axes[0].set_xlabel("Pop Time (sec)")
axes[0].set_title(f"Best Pop Time — {YEAR}")
axes[0].invert_yaxis()

# CS above average
top_cs = df_ct.nlargest(15, "caught_stealing_above_average")
axes[1].barh(top_cs["player_name"], top_cs["caught_stealing_above_average"],
             color=sns.color_palette("crest", 15))
axes[1].set_xlabel("CS Above Average")
axes[1].set_title(f"Top 15 CS Above Average — {YEAR}")
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 12. Catcher Stance
# One-knee vs traditional stance: framing, blocking, throwing impact.

# %%
from savant_extras import catcher_stance

df_cs = catcher_stance(YEAR)
df_cs = coerce_numeric(df_cs)
print(f"Catcher Stance: {len(df_cs)} catchers")
df_cs.head()

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_cs["knee_down_pct"], df_cs["catching_rv"], alpha=0.6, s=30, color="teal")
ax.set_xlabel("Knee Down %")
ax.set_ylabel("Catching Run Value")
ax.set_title(f"Knee Down Rate vs Catching Value — {YEAR}")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 13. Baserunning Run Value
# Total baserunning value (extra bases + stolen bases).

# %%
from savant_extras import baserunning

df_br = baserunning(YEAR)
df_br = coerce_numeric(df_br)
print(f"Baserunning: {len(df_br)} runners")
df_br.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top_br = df_br.nlargest(TOP_N, "runner_runs_tot")
ax.barh(top_br["entity_name"], top_br["runner_runs_tot"],
        color=sns.color_palette("viridis", TOP_N))
ax.set_xlabel("Total Baserunning Run Value")
ax.set_title(f"Top {TOP_N} Baserunners — {YEAR}")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 14. Basestealing Run Value
# Stolen base run value, success rate, lead distances.

# %%
from savant_extras import basestealing

df_bs = basestealing(YEAR)
df_bs = coerce_numeric(df_bs)
print(f"Basestealing: {len(df_bs)} runners")
df_bs.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top_bs = df_bs.nlargest(TOP_N, "runs_stolen_on_running_act")
ax.barh(top_bs["player_name"], top_bs["runs_stolen_on_running_act"],
        color=sns.color_palette("rocket", TOP_N))
ax.set_xlabel("Basestealing Run Value")
ax.set_title(f"Top {TOP_N} Base Stealers — {YEAR}")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 15. Timer Infractions (2023+)
# Pitch clock violations by type.

# %%
from savant_extras import timer_infractions

df_ti = timer_infractions(YEAR)
df_ti = coerce_numeric(df_ti)
print(f"Timer Infractions: {len(df_ti)} players")
df_ti.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Most violations
top_ti = df_ti.nlargest(TOP_N, "all_violations")
axes[0].barh(top_ti["entity_name"], top_ti["all_violations"],
             color=sns.color_palette("Reds", TOP_N))
axes[0].set_xlabel("Total Violations")
axes[0].set_title(f"Most Pitch Timer Violations — {YEAR}")
axes[0].invert_yaxis()

# Violation type breakdown
violation_types = ["pitcher_timer", "batter_timer", "batter_timeout", "catcher_timer", "defensive_shift"]
violation_labels = ["Pitcher", "Batter", "Batter TO", "Catcher", "Def Shift"]
totals = [df_ti[v].sum() for v in violation_types]
axes[1].bar(violation_labels, totals, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"],
            edgecolor="white", width=0.6)
axes[1].set_ylabel("Total Violations")
axes[1].set_title(f"Violations by Type — {YEAR}")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 16. Year-to-Year Changes
# xwOBA changes across seasons.

# %%
from savant_extras import year_to_year

df_yty = year_to_year(YEAR)
df_yty = coerce_numeric(df_yty)
print(f"Year to Year: {len(df_yty)} batters")
df_yty.head()

# %%
if "delta_2023_2024" in df_yty.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    delta = df_yty["delta_2023_2024"].dropna()
    ax.hist(delta, bins=30, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("xwOBA Change (2023 → 2024)")
    ax.set_ylabel("Count")
    ax.set_title(f"Year-to-Year xwOBA Change Distribution — {YEAR}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ## Save All Data as CSV
# Export all leaderboards for the dataset.

# %%
import os
output_dir = "savant_extras_2024"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "bat_tracking": df_bat,
    "pitch_tempo": df_tempo,
    "arm_strength": df_arm,
    "batted_ball": df_bb,
    "home_runs": df_hr,
    "pitch_movement": df_pm,
    "swing_take": df_st,
    "pitcher_arm_angle": df_angle,
    "running_game": df_rg,
    "catcher_blocking": df_cb,
    "catcher_throwing": df_ct,
    "catcher_stance": df_cs,
    "baserunning": df_br,
    "basestealing": df_bs,
    "timer_infractions": df_ti,
    "year_to_year": df_yty,
}

for name, df in datasets.items():
    path = f"{output_dir}/{name}_2024.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}: {len(df)} rows x {len(df.columns)} cols")

print(f"\nTotal: {len(datasets)} CSV files saved to {output_dir}/")

# %% [markdown]
# ---
# ## Summary
#
# **savant-extras** provides 33 functions covering 16 Baseball Savant leaderboards:
#
# | Category | Leaderboards |
# |---|---|
# | Batting | Bat Tracking, Batted Ball, Home Runs, Swing & Take, Year-to-Year |
# | Pitching | Pitch Tempo, Pitch Movement, Arm Angle, Running Game, Timer Infractions |
# | Catching | Blocking, Throwing, Stance |
# | Baserunning | Baserunning Run Value, Basestealing Run Value |
# | Fielding | Arm Strength |
#
# Install: `pip install savant-extras`
#
# - **PyPI**: [savant-extras](https://pypi.org/project/savant-extras/)
# - **GitHub**: [yasumorishima/savant-extras](https://github.com/yasumorishima/savant-extras)
# - **Demo App**: [MLB Bat Tracking Dashboard](https://yasumorishima-mlb-bat-tracking.streamlit.app/)
