# %% [markdown]
# # savant-extras: All Baseball Savant Leaderboards in One Package
#
# **[savant-extras](https://github.com/yasumorishima/savant-extras)** provides 20+ Baseball Savant leaderboards that [pybaseball](https://github.com/jldbc/pybaseball) doesn't support — all as simple one-line function calls returning DataFrames.
#
# This notebook demonstrates every leaderboard with visualizations using **2024–2025 season data**.
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

YEARS = [2024, 2025]
YEAR = 2025   # used for single-year visualizations
TOP_N = 20    # top N players for bar charts

def coerce_numeric(df):
    """Convert columns that look numeric but are stored as object dtype."""
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > df[col].notna().sum() * 0.5:
                df[col] = converted
    return df

def fetch_years(fn, year_col="year", sleep_sec=1.0, **kwargs):
    """Call fn(year, **kwargs) for each year in YEARS and concatenate."""
    frames = []
    for i, y in enumerate(YEARS):
        if i > 0:
            time.sleep(sleep_sec)
        df = fn(y, **kwargs)
        df = coerce_numeric(df)
        if year_col not in df.columns:
            df[year_col] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# %% [markdown]
# ---
# ## 1. Bat Tracking (2024+)
# Bat speed, attack angle, swing tilt — with custom date ranges.

# %%
from savant_extras import bat_tracking

bat_frames = []
for y in YEARS:
    df_tmp = bat_tracking(f"{y}-04-01", f"{y}-09-30", min_swings=100)
    df_tmp = coerce_numeric(df_tmp)
    df_tmp["year"] = y
    bat_frames.append(df_tmp)
df_bat = pd.concat(bat_frames, ignore_index=True)
print(f"Bat Tracking: {len(df_bat)} player-seasons")
df_bat.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top = df_bat[df_bat["year"] == YEAR].nlargest(TOP_N, "avg_bat_speed")
ax.barh(top["name"], top["avg_bat_speed"], color=sns.color_palette("rocket", TOP_N))
ax.set_xlabel("Average Bat Speed (mph)", fontsize=14)
ax.set_title(f"Top {TOP_N} Bat Speed — {YEAR}", fontsize=16)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_bat["avg_bat_speed"], df_bat["attack_angle"], alpha=0.4, s=15, c=df_bat["year"], cmap="coolwarm")
ax.set_xlabel("Average Bat Speed (mph)", fontsize=14)
ax.set_ylabel("Attack Angle (°)", fontsize=14)
ax.set_title(f"Bat Speed vs Attack Angle — 2024–2025", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 2. Pitch Tempo (2010+)
# Pace metrics — median seconds between pitches, hot/warm/cold frequency.

# %%
from savant_extras import pitch_tempo

df_tempo = fetch_years(pitch_tempo)
print(f"Pitch Tempo: {len(df_tempo)} pitcher-seasons")
df_tempo.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fastest = df_tempo[df_tempo["year"] == YEAR].nsmallest(TOP_N, "median_seconds_empty")
axes[0].barh(fastest["entity_name"], fastest["median_seconds_empty"],
             color=sns.color_palette("YlOrRd_r", TOP_N))
axes[0].set_xlabel("Median Seconds (Bases Empty)", fontsize=14)
axes[0].set_title(f"Fastest Tempo — {YEAR}", fontsize=16)
axes[0].invert_yaxis()

axes[1].hist(df_tempo[df_tempo["year"] == YEAR]["median_seconds_empty"].dropna(),
             bins=30, color="steelblue", edgecolor="white")
axes[1].set_xlabel("Median Seconds (Bases Empty)", fontsize=14)
axes[1].set_ylabel("Count", fontsize=14)
axes[1].set_title(f"Pitch Tempo Distribution — {YEAR}", fontsize=16)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 3. Arm Strength (2020+)
# Fielder throw speed by position.

# %%
from savant_extras import arm_strength

df_arm = fetch_years(arm_strength, min_throws=50)
print(f"Arm Strength: {len(df_arm)} fielder-seasons")
df_arm.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

top_arm = df_arm[df_arm["year"] == YEAR].nlargest(TOP_N, "max_arm_strength")
axes[0].barh(top_arm["fielder_name"], top_arm["max_arm_strength"],
             color=sns.color_palette("flare", TOP_N))
axes[0].set_xlabel("Max Arm Strength (mph)", fontsize=14)
axes[0].set_title(f"Top {TOP_N} Max Throw Speed — {YEAR}", fontsize=16)
axes[0].invert_yaxis()

pos_order = ["RF", "CF", "LF", "SS", "3B", "2B", "1B"]
pos_data = df_arm[(df_arm["year"] == YEAR) & df_arm["primary_position"].isin(pos_order)]
sns.boxplot(data=pos_data, x="primary_position", y="arm_overall",
            order=pos_order, ax=axes[1], palette="Set2")
axes[1].set_xlabel("Position", fontsize=14)
axes[1].set_ylabel("Arm Strength Overall (mph)", fontsize=14)
axes[1].set_title(f"Arm Strength by Position — {YEAR}", fontsize=16)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 4. Batted Ball Profile
# Ground ball, fly ball, line drive rates and pull/oppo splits.

# %%
from savant_extras import batted_ball

df_bb = fetch_years(batted_ball)
print(f"Batted Ball: {len(df_bb)} batter-seasons")
df_bb.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bb25 = df_bb[df_bb["year"] == YEAR]
axes[0].scatter(bb25["gb_rate"], bb25["fb_rate"], alpha=0.5, s=20, color="teal")
axes[0].set_xlabel("Ground Ball Rate", fontsize=14)
axes[0].set_ylabel("Fly Ball Rate", fontsize=14)
axes[0].set_title(f"GB Rate vs FB Rate — {YEAR}", fontsize=16)

axes[1].hist(bb25["pull_rate"].dropna(), bins=25, color="coral", edgecolor="white")
axes[1].set_xlabel("Pull Rate", fontsize=14)
axes[1].set_ylabel("Count", fontsize=14)
axes[1].set_title(f"Pull Rate Distribution — {YEAR}", fontsize=16)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 5. Home Runs
# HR distance, exit velocity, expected HR, no-doubters.

# %%
from savant_extras import home_runs

df_hr = fetch_years(home_runs)
print(f"Home Runs: {len(df_hr)} batter-seasons")
df_hr.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

hr25 = df_hr[df_hr["year"] == YEAR]
top_nd = hr25.nlargest(TOP_N, "no_doubters")
axes[0].barh(top_nd["player"], top_nd["no_doubters"],
             color=sns.color_palette("magma", TOP_N))
axes[0].set_xlabel("No-Doubter Home Runs", fontsize=14)
axes[0].set_title(f"Top {TOP_N} No-Doubters — {YEAR}", fontsize=16)
axes[0].invert_yaxis()

axes[1].scatter(hr25["xhr"], hr25["hr_total"], alpha=0.5, s=20, color="purple")
lims = [0, hr25[["xhr", "hr_total"]].max().max() + 5]
axes[1].plot(lims, lims, "--", color="gray", alpha=0.5)
axes[1].set_xlabel("Expected HR (xHR)", fontsize=14)
axes[1].set_ylabel("Actual HR", fontsize=14)
axes[1].set_title(f"HR vs xHR — {YEAR}", fontsize=16)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 6. Pitch Movement
# Horizontal and vertical break by pitch type.

# %%
from savant_extras import pitch_movement

df_pm = fetch_years(pitch_movement)
print(f"Pitch Movement: {len(df_pm)} pitcher-pitch-season combos")
df_pm.head()

# %%
fig, ax = plt.subplots(figsize=(10, 8))
pitch_types = ["FF", "SL", "CU", "CH", "SI", "FC", "ST", "SV"]
colors = sns.color_palette("Set1", len(pitch_types))
pm25 = df_pm[df_pm["year"] == YEAR]

for pt, color in zip(pitch_types, colors):
    sub = pm25[pm25["pitch_type"] == pt]
    if len(sub) > 0:
        ax.scatter(sub["pitcher_break_x"], sub["pitcher_break_z"],
                   alpha=0.3, s=10, color=color, label=pt)

ax.set_xlabel("Horizontal Break (in)", fontsize=14)
ax.set_ylabel("Vertical Break (in)", fontsize=14)
ax.set_title(f"Pitch Movement by Type — {YEAR}", fontsize=16)
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

df_st = fetch_years(swing_take)
df_st = coerce_numeric(df_st)
print(f"Swing & Take: {len(df_st)} batter-seasons")
df_st.head()

# %%
st25 = df_st[df_st["year"] == YEAR]
if len(st25) > 0 and "runs_all" in st25.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    top_st = st25.nlargest(TOP_N, "runs_all")
    ax.barh(top_st["last_name, first_name"], top_st["runs_all"],
            color=sns.color_palette("viridis", TOP_N))
    ax.set_xlabel("Total Run Value (Swing + Take)", fontsize=14)
    ax.set_title(f"Top {TOP_N} Swing & Take Run Value — {YEAR}", fontsize=16)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print("swing_take: no data available (known issue with Baseball Savant API)")

# %% [markdown]
# ---
# ## 8. Pitcher Arm Angle
# Release point angles and positions.

# %%
from savant_extras import pitcher_arm_angle

df_angle = fetch_years(pitcher_arm_angle)
print(f"Pitcher Arm Angle: {len(df_angle)} pitcher-seasons")
df_angle.head()

# %%
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(df_angle[df_angle["year"] == YEAR]["ball_angle"].dropna(),
        bins=30, color="dodgerblue", edgecolor="white")
ax.set_xlabel("Ball Angle (°)", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
ax.set_title(f"Pitcher Arm Angle Distribution — {YEAR}", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 9. Running Game (Pitcher)
# Pitcher's ability to control the running game.

# %%
from savant_extras import running_game

df_rg = fetch_years(running_game)
print(f"Running Game: {len(df_rg)} pitcher-seasons")
df_rg.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top_rg = df_rg[df_rg["year"] == YEAR].nlargest(TOP_N, "runs_prevented_on_running_attr")
ax.barh(top_rg["player_name"], top_rg["runs_prevented_on_running_attr"],
        color=sns.color_palette("crest", TOP_N))
ax.set_xlabel("Runs Prevented on Running", fontsize=14)
ax.set_title(f"Top {TOP_N} Pitchers — Running Game — {YEAR}", fontsize=16)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 10. Catcher Blocking
# Blocks above average, PB/WP prevention.

# %%
from savant_extras import catcher_blocking

df_cb = fetch_years(catcher_blocking)
print(f"Catcher Blocking: {len(df_cb)} catcher-seasons")
df_cb.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top_cb = df_cb[df_cb["year"] == YEAR].nlargest(15, "blocks_above_average")
ax.barh(top_cb["player_name"], top_cb["blocks_above_average"],
        color=sns.color_palette("mako", 15))
ax.set_xlabel("Blocks Above Average", fontsize=14)
ax.set_title(f"Top 15 Catchers — Blocking — {YEAR}", fontsize=16)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 11. Catcher Throwing
# Pop time, exchange time, CS rate, arm strength.

# %%
from savant_extras import catcher_throwing

df_ct = fetch_years(catcher_throwing)
print(f"Catcher Throwing: {len(df_ct)} catcher-seasons")
df_ct.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ct25 = df_ct[df_ct["year"] == YEAR]
top_pop = ct25.nsmallest(15, "pop_time")
axes[0].barh(top_pop["player_name"], top_pop["pop_time"],
             color=sns.color_palette("YlOrRd_r", 15))
axes[0].set_xlabel("Pop Time (sec)", fontsize=14)
axes[0].set_title(f"Best Pop Time — {YEAR}", fontsize=16)
axes[0].invert_yaxis()

top_cs = ct25.nlargest(15, "caught_stealing_above_average")
axes[1].barh(top_cs["player_name"], top_cs["caught_stealing_above_average"],
             color=sns.color_palette("crest", 15))
axes[1].set_xlabel("CS Above Average", fontsize=14)
axes[1].set_title(f"Top 15 CS Above Average — {YEAR}", fontsize=16)
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 12. Catcher Stance
# One-knee vs traditional stance: framing, blocking, throwing impact.

# %%
from savant_extras import catcher_stance

df_cs = fetch_years(catcher_stance)
print(f"Catcher Stance: {len(df_cs)} catcher-seasons")
df_cs.head()

# %%
fig, ax = plt.subplots(figsize=(8, 6))
cs25 = df_cs[df_cs["year"] == YEAR]
ax.scatter(cs25["knee_down_pct"], cs25["catching_rv"], alpha=0.6, s=30, color="teal")
ax.set_xlabel("Knee Down %", fontsize=14)
ax.set_ylabel("Catching Run Value", fontsize=14)
ax.set_title(f"Knee Down Rate vs Catching Value — {YEAR}", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 13. Baserunning Run Value
# Total baserunning value (extra bases + stolen bases).

# %%
from savant_extras import baserunning

df_br = fetch_years(baserunning)
print(f"Baserunning: {len(df_br)} runner-seasons")
df_br.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top_br = df_br[df_br["year"] == YEAR].nlargest(TOP_N, "runner_runs_tot")
ax.barh(top_br["entity_name"], top_br["runner_runs_tot"],
        color=sns.color_palette("viridis", TOP_N))
ax.set_xlabel("Total Baserunning Run Value", fontsize=14)
ax.set_title(f"Top {TOP_N} Baserunners — {YEAR}", fontsize=16)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 14. Basestealing Run Value
# Stolen base run value, success rate, lead distances.

# %%
from savant_extras import basestealing

df_bs = fetch_years(basestealing)
print(f"Basestealing: {len(df_bs)} runner-seasons")
df_bs.head()

# %%
fig, ax = plt.subplots(figsize=(10, 6))
top_bs = df_bs[df_bs["year"] == YEAR].nlargest(TOP_N, "runs_stolen_on_running_act")
ax.barh(top_bs["player_name"], top_bs["runs_stolen_on_running_act"],
        color=sns.color_palette("rocket", TOP_N))
ax.set_xlabel("Basestealing Run Value", fontsize=14)
ax.set_title(f"Top {TOP_N} Base Stealers — {YEAR}", fontsize=16)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 15. Timer Infractions (2023+)
# Pitch clock violations by type.

# %%
from savant_extras import timer_infractions

df_ti = fetch_years(timer_infractions)
print(f"Timer Infractions: {len(df_ti)} player-seasons")
df_ti.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ti25 = df_ti[df_ti["year"] == YEAR]
top_ti = ti25.nlargest(TOP_N, "all_violations")
axes[0].barh(top_ti["entity_name"], top_ti["all_violations"],
             color=sns.color_palette("Reds", TOP_N))
axes[0].set_xlabel("Total Violations", fontsize=14)
axes[0].set_title(f"Most Pitch Timer Violations — {YEAR}", fontsize=16)
axes[0].invert_yaxis()

violation_types = ["pitcher_timer", "batter_timer", "batter_timeout", "catcher_timer", "defensive_shift"]
violation_labels = ["Pitcher", "Batter", "Batter TO", "Catcher", "Def Shift"]
totals = [ti25[v].sum() for v in violation_types]
axes[1].bar(violation_labels, totals,
            color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"],
            edgecolor="white", width=0.6)
axes[1].set_ylabel("Total Violations", fontsize=14)
axes[1].set_title(f"Violations by Type — {YEAR}", fontsize=16)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 16. Year-to-Year Changes
# xwOBA changes across seasons.

# %%
from savant_extras import year_to_year

df_yty = fetch_years(year_to_year)
print(f"Year to Year: {len(df_yty)} batter-seasons")
df_yty.head()

# %%
delta_col = "delta_2024_2025" if "delta_2024_2025" in df_yty.columns else "delta_2023_2024"
if delta_col in df_yty.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    delta = df_yty[delta_col].dropna()
    ax.hist(delta, bins=30, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel(f"xwOBA Change ({delta_col.replace('delta_', '').replace('_', ' → ')})", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(f"Year-to-Year xwOBA Change Distribution", fontsize=16)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ---
# ## 17. Park Factors (FanGraphs)
# Per-team park factors for runs, HR, 1B/2B/3B, SO, BB, FIP. 100 = neutral.

# %%
from savant_extras import park_factors_range

try:
    df_pf = park_factors_range(YEARS[0], YEARS[-1])
    print(f"Park Factors: {len(df_pf)} team-seasons")
    df_pf.head()
except Exception as e:
    print(f"park_factors: skipped ({e})")
    df_pf = pd.DataFrame()

# %%
if not df_pf.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pf25 = df_pf[df_pf["season"] == YEAR].sort_values("pf_5yr", ascending=True)
    colors = ["#e74c3c" if v > 100 else "#3498db" for v in pf25["pf_5yr"]]
    axes[0].barh(pf25["team"], pf25["pf_5yr"], color=colors)
    axes[0].axvline(100, color="gray", lw=1.5, ls="--")
    axes[0].set_xlabel("5-Year Park Factor (runs)", fontsize=14)
    axes[0].set_title(f"Park Factors — {YEAR}", fontsize=16)
    axes[1].scatter(pf25["pf_5yr"], pf25["pf_hr"], alpha=0.7, s=50, color="steelblue")
    for _, row in pf25.iterrows():
        axes[1].annotate(row["team"], (row["pf_5yr"], row["pf_hr"]),
                         fontsize=7, ha="center", va="bottom")
    axes[1].axhline(100, color="gray", lw=0.8, ls="--")
    axes[1].axvline(100, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("5-Year Park Factor (runs)", fontsize=14)
    axes[1].set_ylabel("HR Park Factor", fontsize=14)
    axes[1].set_title(f"Runs vs HR Park Factor — {YEAR}", fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("Park Factors: skipped (FanGraphs may block cloud IPs)")

# %% [markdown]
# ---
# ## 18. Outs Above Average (OAA)
# Defensive runs saved vs expected, with directional breakdowns.

# %%
from pybaseball import statcast_outs_above_average

def _oaa(year, **_):
    return statcast_outs_above_average(year, 'all')

df_oaa = fetch_years(_oaa, year_col="year")
print(f"Outs Above Average: {len(df_oaa)} fielder-seasons")
df_oaa.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

oaa25 = df_oaa[df_oaa["year"] == YEAR]
top_oaa = oaa25.nlargest(TOP_N, "outs_above_average")
axes[0].barh(top_oaa["last_name, first_name"], top_oaa["outs_above_average"],
             color=sns.color_palette("crest", TOP_N))
axes[0].set_xlabel("Outs Above Average", fontsize=14)
axes[0].set_title(f"Top {TOP_N} Defenders by OAA — {YEAR}", fontsize=16)
axes[0].invert_yaxis()

if "fielding_runs_prevented" in oaa25.columns:
    axes[1].scatter(oaa25["outs_above_average"], oaa25["fielding_runs_prevented"],
                    alpha=0.5, s=25, color="teal")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("Outs Above Average", fontsize=14)
    axes[1].set_ylabel("Fielding Runs Prevented", fontsize=14)
    axes[1].set_title(f"OAA vs Fielding Runs — {YEAR}", fontsize=16)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 19. Outfield Jump
# First-step reaction and routing efficiency for outfielders (2-star plays only).

# %%
from pybaseball import statcast_outfielder_jump

df_oj = fetch_years(statcast_outfielder_jump, year_col="year")
print(f"Outfield Jump: {len(df_oj)} outfielder-seasons")
df_oj.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

oj25 = df_oj[df_oj["year"] == YEAR]
name_col = "player_name" if "player_name" in oj25.columns else oj25.columns[0]

if "rel_league_bootup_distance" in oj25.columns:
    top_oj = oj25.nlargest(TOP_N, "rel_league_bootup_distance")
    axes[0].barh(top_oj[name_col], top_oj["rel_league_bootup_distance"],
                 color=sns.color_palette("rocket", TOP_N))
    axes[0].set_xlabel("Jump vs League Avg (ft)", fontsize=14)
    axes[0].set_title(f"Top {TOP_N} Outfield Jump — {YEAR}", fontsize=16)
    axes[0].invert_yaxis()

if "rel_league_reaction_distance" in oj25.columns and "rel_league_routing_distance" in oj25.columns:
    axes[1].scatter(oj25["rel_league_reaction_distance"], oj25["rel_league_routing_distance"],
                    alpha=0.5, s=25, color="darkorange")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("Reaction Distance vs Avg (ft)", fontsize=14)
    axes[1].set_ylabel("Routing Distance vs Avg (ft)", fontsize=14)
    axes[1].set_title(f"First Step vs Route Efficiency — {YEAR}", fontsize=16)

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 20. Pitcher Quality — Stuff+ / Location+ / Pitching+ (FanGraphs)
# Model-based pitcher quality metrics. 100 = MLB average.

# %%
from pybaseball import fg_pitching_data

_PQ_COLS = ["Name", "Team", "Age", "IP", "Stuff+", "Location+", "Pitching+"]
_PQ_RENAME = {"Name": "name", "Team": "team", "Age": "age", "IP": "ip",
              "Stuff+": "stuff_plus", "Location+": "location_plus", "Pitching+": "pitching_plus"}
pq_frames = []
for y in YEARS:
    try:
        df_tmp = fg_pitching_data(y, qual=0)
        avail = [c for c in _PQ_COLS if c in df_tmp.columns]
        df_tmp = df_tmp[avail].rename(columns=_PQ_RENAME).copy()
        df_tmp = coerce_numeric(df_tmp)
        df_tmp["season"] = y
        pq_frames.append(df_tmp)
        print(f"pitcher_quality {y}: {len(df_tmp)} pitchers")
    except Exception as e:
        print(f"pitcher_quality {y}: skipped ({e})")
    time.sleep(1.5)
df_pq = pd.concat(pq_frames, ignore_index=True) if pq_frames else pd.DataFrame()
print(f"Pitcher Quality total: {len(df_pq)} pitcher-seasons")
df_pq.head()

# %%
if not df_pq.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pq25 = df_pq[df_pq["season"] == YEAR]
    if "pitching_plus" in pq25.columns:
        top_pq = pq25.nlargest(TOP_N, "pitching_plus")
        axes[0].barh(top_pq["name"], top_pq["pitching_plus"],
                     color=sns.color_palette("flare", TOP_N))
        axes[0].axvline(100, color="gray", lw=1.5, ls="--")
        axes[0].set_xlabel("Pitching+ (100 = MLB avg)", fontsize=14)
        axes[0].set_title(f"Top {TOP_N} Pitching+ — {YEAR}", fontsize=16)
        axes[0].invert_yaxis()
    if "stuff_plus" in pq25.columns and "location_plus" in pq25.columns:
        axes[1].scatter(pq25["stuff_plus"], pq25["location_plus"],
                        alpha=0.5, s=20, color="purple")
        axes[1].axhline(100, color="gray", lw=0.8, ls="--")
        axes[1].axvline(100, color="gray", lw=0.8, ls="--")
        axes[1].set_xlabel("Stuff+ (100 = avg)", fontsize=14)
        axes[1].set_ylabel("Location+ (100 = avg)", fontsize=14)
        axes[1].set_title(f"Stuff+ vs Location+ — {YEAR}", fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print("Pitcher Quality: skipped (FanGraphs may block cloud IPs)")

# %% [markdown]
# ---
# ## Save All Data as CSV (2024–2025)
# Export all leaderboards for the dataset. Files named `*_2024_2025.csv`.

# %%
import os
output_dir = "savant_extras_2024_2025"
os.makedirs(output_dir, exist_ok=True)

# park_factors uses "season" column; pitcher_quality uses "season" column
# all others have "year" column added by fetch_years()
datasets = {
    "bat_tracking":       df_bat,
    "pitch_tempo":        df_tempo,
    "arm_strength":       df_arm,
    "batted_ball":        df_bb,
    "home_runs":          df_hr,
    "pitch_movement":     df_pm,
    "swing_take":         df_st,
    "pitcher_arm_angle":  df_angle,
    "running_game":       df_rg,
    "catcher_blocking":   df_cb,
    "catcher_throwing":   df_ct,
    "catcher_stance":     df_cs,
    "baserunning":        df_br,
    "basestealing":       df_bs,
    "timer_infractions":  df_ti,
    "year_to_year":       df_yty,
    "park_factors":       df_pf,
    "outs_above_average": df_oaa,
    "outfield_jump":      df_oj,
    "pitcher_quality":    df_pq,
}

for name, df in datasets.items():
    path = f"{output_dir}/{name}_2024_2025.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}: {len(df)} rows x {len(df.columns)} cols")

print(f"\nTotal: {len(datasets)} CSV files saved to {output_dir}/")

# %% [markdown]
# ---
# ## Summary
#
# **savant-extras** provides 21+ leaderboards (pybaseball doesn't support any of them):
#
# | Category | Leaderboards |
# |---|---|
# | Batting | Bat Tracking, Batted Ball, Home Runs, Swing & Take, Year-to-Year |
# | Pitching | Pitch Tempo, Pitch Movement, Arm Angle, Running Game, Timer Infractions, Pitcher Quality (Stuff+) |
# | Catching | Blocking, Throwing, Stance |
# | Baserunning | Baserunning Run Value, Basestealing Run Value |
# | Fielding | Arm Strength, Outs Above Average, Outfield Jump |
# | Park | Park Factors (FanGraphs) |
#
# Install: `pip install savant-extras`
#
# - **PyPI**: [savant-extras](https://pypi.org/project/savant-extras/)
# - **GitHub**: [yasumorishima/savant-extras](https://github.com/yasumorishima/savant-extras)
# - **Dataset**: [Baseball Savant Leaderboards 2024-2025](https://www.kaggle.com/datasets/yasunorim/baseball-savant-leaderboards-2024)
