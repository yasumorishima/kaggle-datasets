# %% [markdown]
# # savant-extras: All Baseball Savant Leaderboards in One Package
#
# **[savant-extras](https://github.com/yasumorishima/savant-extras)** turns Baseball Savant leaderboards that [pybaseball](https://github.com/jldbc/pybaseball) doesn't cover into one-line function calls returning DataFrames: bat tracking, pitch tempo, arm strength, catcher stance, baserunning, park factors, **ABS challenges** and **Triple-A pitch-level Statcast**.
#
# This notebook calls every leaderboard live for the **2024, 2025 and 2026 regular seasons** (2026 ended on September 27) and draws one chart from each.
#
# ```
# pip install "savant-extras>=0.6.0"
# ```
#
# **Why 0.6.0 matters.** Baseball Savant ignores query parameters it does not recognise and then answers with *the current season*. Up to 0.5.0 several functions sent parameter names Savant no longer reads, so asking for 2024 quietly returned 2026. 0.6.0 sends the parameters Savant reads and checks the season it gets back. Section 0 shows how to check this yourself.
#
# The same leaderboards are published as CSVs, with build-time checks, in the dataset **[Baseball Savant Leaderboards (2024-2026)](https://www.kaggle.com/datasets/yasunorim/baseball-savant-leaderboards-2024)**, attached to this notebook.

# %% [markdown]
# ## Setup

# %%
!pip install -q "savant-extras>=0.6.0"

import io
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import savant_extras as sx

sns.set_theme(style="whitegrid")
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12})
warnings.filterwarnings("ignore", category=FutureWarning)
print("savant-extras", sx.__version__)

YEARS = [2024, 2025, 2026]
YEAR = 2026   # single-season charts
TOP_N = 15


def fetch_years(fn, years=YEARS, sleep_sec=1.0, **kwargs):
    """Call fn(year, **kwargs) for each season and stack the results with a `year` column.

    `year` is always written from the request: some Savant tables return an empty `year`
    column, and some have none.
    """
    frames = []
    for i, y in enumerate(years):
        if i:
            time.sleep(sleep_sec)
        df = fn(y, **kwargs)
        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def savant_csv(url):
    """A Savant leaderboard CSV endpoint that savant-extras does not wrap."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))


def barh(ax, df, label, value, title, xlabel, palette="crest", n=TOP_N, smallest=False):
    top = (df.nsmallest if smallest else df.nlargest)(n, value)
    ax.barh(top[label], top[value], color=sns.color_palette(palette, len(top)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

# %% [markdown]
# ---
# ## 0. Is each season really a different season?
#
# A leaderboard that silently returned the current season would give the same table for every year. A quick check: drop the columns that only name the season and compare what is left.

# %%
SEASON_COLS = ["year", "start_year", "end_year"]

def same_table(df, a, b, key):
    """True when seasons a and b hold the same rows once the columns naming the season are dropped."""
    def part(y):
        d = df[df.year == y].drop(columns=[c for c in SEASON_COLS if c in df.columns])
        return d.sort_values(key).reset_index(drop=True)
    x, y = part(a), part(b)
    return x.shape == y.shape and x.equals(y)

df_cb = fetch_years(sx.catcher_blocking)
for a, b in [(2024, 2025), (2025, 2026), (2024, 2026)]:
    print(f"catcher_blocking {a} vs {b}: same table = {same_table(df_cb, a, b, 'player_id')}")
df_cb.groupby("year").size()

# %% [markdown]
# ---
# ## 1. Bat Tracking (2024+)
# Bat speed, attack angle and swing tilt over any date range. Here each regular season, minimum 100 competitive swings.

# %%
SEASONS = {2024: ("2024-03-20", "2024-09-30"), 2025: ("2025-03-18", "2025-09-28"), 2026: ("2026-03-25", "2026-09-27")}
frames = []
for y, (start, end) in SEASONS.items():
    d = sx.bat_tracking(start, end, min_swings=100)
    d["year"] = y
    frames.append(d)
    time.sleep(1)
df_bat = pd.concat(frames, ignore_index=True)
print(df_bat.groupby("year").size())
df_bat.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
barh(axes[0], df_bat[df_bat.year == YEAR], "name", "avg_bat_speed",
     f"Fastest average bat speed, {YEAR}", "Average bat speed (mph)", "rocket")
axes[0].set_xlim(df_bat.avg_bat_speed.quantile(0.5), None)
sns.kdeplot(data=df_bat, x="avg_bat_speed", hue="year", ax=axes[1], palette="viridis", common_norm=False)
axes[1].set_title("Bat speed by season")
axes[1].set_xlabel("Average bat speed (mph)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 2. Pitch Tempo (2010+)
# Median seconds between pitches, bases empty.

# %%
df_tempo = fetch_years(sx.pitch_tempo)
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.boxplot(data=df_tempo, x="year", y="median_seconds_empty", ax=ax, color="#4C72B0")
ax.set_title("Seconds between pitches, bases empty")
ax.set_xlabel("")
ax.set_ylabel("Median seconds")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 3. Arm Strength (2020+)
# Fielder throw speed by position.

# %%
df_arm = fetch_years(sx.arm_strength)
# primary_position is the scorer's position number
POS = {3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}
pos_order = ["RF", "CF", "LF", "SS", "3B", "2B", "1B"]
d = df_arm[df_arm.year == YEAR].assign(primary_position=lambda x: x.primary_position.map(POS)).dropna(subset=["primary_position"])
fig, ax = plt.subplots(figsize=(10, 5.5))
sns.boxplot(data=d, x="primary_position", y="arm_overall", order=pos_order, ax=ax, color="#55A868")
ax.set_title(f"Arm strength by position, {YEAR}")
ax.set_xlabel("Position")
ax.set_ylabel("Arm strength (mph)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 4. Batted Ball Profile
# Ground ball, fly ball, line drive rates and pull / straight / oppo splits.

# %%
df_bb = fetch_years(sx.batted_ball)
d = df_bb[df_bb.year == YEAR]
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(d.pull_air_rate, d.gb_rate, alpha=0.6, s=25, color="teal")
ax.set_title(f"Pulled fly balls vs ground balls, {YEAR}")
ax.set_xlabel("Pulled air-ball rate")
ax.set_ylabel("Ground-ball rate")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 5. Home Runs
# HR totals against park-adjusted expected HR, and no-doubters.

# %%
df_hr = fetch_years(sx.home_runs)
d = df_hr[df_hr.year == YEAR]
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
barh(axes[0], d, "player", "no_doubters", f"Most no-doubter home runs, {YEAR}", "No-doubters", "magma")
axes[1].scatter(d.xhr, d.hr_total, alpha=0.5, s=20, color="purple")
lim = d[["xhr", "hr_total"]].max().max() + 3
axes[1].plot([0, lim], [0, lim], "--", color="gray")
axes[1].set_title(f"Home runs vs expected, {YEAR}")
axes[1].set_xlabel("Expected HR (park-adjusted)")
axes[1].set_ylabel("Actual HR")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 6. Pitch Movement
# Savant serves **one pitch type per request** (an empty type returns four-seamers only), so loop over the types.

# %%
PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "CH", "FS"]
frames = []
for pt in PITCH_TYPES:
    d = sx.pitch_movement(YEAR, pitch_type=pt)
    if len(d):
        frames.append(d)
    time.sleep(1)
df_pm = pd.concat(frames, ignore_index=True)
print(df_pm.pitch_type.value_counts())

fig, ax = plt.subplots(figsize=(10, 8))
for pt, color in zip(PITCH_TYPES, sns.color_palette("tab10", len(PITCH_TYPES))):
    s = df_pm[df_pm.pitch_type == pt]
    ax.scatter(s.pitcher_break_x, s.pitcher_break_z, s=10, alpha=0.4, color=color, label=pt)
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_title(f"Pitch movement by type, {YEAR}")
ax.set_xlabel("Horizontal break (in)")
ax.set_ylabel("Vertical break, with gravity (in)")
ax.legend(title="Pitch type", markerscale=3, ncol=3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 7. Swing & Take Run Value
# Run value by attack zone. Savant's default list is the 300 batters who saw the most pitches.

# %%
df_st = fetch_years(sx.swing_take)
d = df_st[df_st.year == YEAR]
fig, ax = plt.subplots(figsize=(10, 6))
barh(ax, d, "last_name, first_name", "runs_all", f"Swing/take run value, {YEAR}", "Runs", "viridis")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 8. Pitcher Arm Angle

# %%
df_angle = fetch_years(sx.pitcher_arm_angle)
fig, ax = plt.subplots(figsize=(9, 5.5))
for (y, g), c in zip(df_angle.groupby("year"), sns.color_palette("viridis", 3)):
    ax.hist(g.ball_angle.dropna(), bins=30, alpha=0.45, color=c, label=str(y))
ax.set_title("Pitcher arm angle")
ax.set_xlabel("Arm angle (degrees)")
ax.set_ylabel("Pitchers")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 9. Running Game (pitchers) and 10. Catcher Throwing

# %%
df_rg = fetch_years(sx.running_game)
df_ct = fetch_years(sx.catcher_throwing)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
barh(axes[0], df_rg[df_rg.year == YEAR], "player_name", "runs_prevented_on_running_attr",
     f"Pitchers: runs prevented on the bases, {YEAR}", "Runs prevented")
barh(axes[1], df_ct[df_ct.year == YEAR], "player_name", "pop_time",
     f"Catchers: fastest pop time, {YEAR}", "Pop time to 2B (s)", "YlOrRd_r", smallest=True)
axes[1].set_xlim(df_ct.pop_time.min() - 0.03, None)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 11. Catcher Blocking and 12. Catcher Stance
# One row per catcher: how often they set up with a knee down, and the value of their blocking.

# %%
df_cs = fetch_years(sx.catcher_stance)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
barh(axes[0], df_cb[df_cb.year == YEAR], "player_name", "blocks_above_average",
     f"Blocks above average, {YEAR}", "Blocks above average", "mako")
d = df_cs[df_cs.year == YEAR]
axes[1].scatter(d.knee_down_pct, d.catching_rv, s=35, alpha=0.7, color="teal")
axes[1].set_title(f"Knee-down rate vs catching run value, {YEAR}")
axes[1].set_xlabel("Share of pitches with a knee down")
axes[1].set_ylabel("Catching run value")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 13. Baserunning and 14. Basestealing Run Value

# %%
df_br = fetch_years(sx.baserunning)
df_bs = fetch_years(sx.basestealing)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
barh(axes[0], df_br[df_br.year == YEAR], "entity_name", "runner_runs_tot",
     f"Baserunning run value, {YEAR}", "Runs", "viridis")
barh(axes[1], df_bs[df_bs.year == YEAR], "player_name", "runs_stolen_on_running_act",
     f"Basestealing run value, {YEAR}", "Runs", "rocket")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 15. Timer Infractions (2023+)
# Pitch clock violations by team.

# %%
df_ti = fetch_years(sx.timer_infractions)
types = ["pitcher_timer", "batter_timer", "batter_timeout", "catcher_timer", "defensive_shift"]
tot = df_ti.groupby("year")[types].sum()
tot.columns = ["Pitcher", "Batter", "Batter timeout", "Catcher", "Defensive shift"]
ax = tot.plot(kind="bar", stacked=True, figsize=(9, 5.5), colormap="tab10", rot=0)
ax.set_title("Pitch clock violations by type")
ax.set_xlabel("")
ax.set_ylabel("Violations")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 16. Year-to-Year xwOBA
# `year` selects the qualified batters; every season from 2015 comes back as a column.

# %%
df_yty = sx.year_to_year(2025)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(df_yty["delta_2025_2026"].dropna(), bins=30, color="steelblue", edgecolor="white")
ax.axvline(0, color="red", ls="--")
ax.set_title("xwOBA change, 2025 to 2026 (2025 qualifiers)")
ax.set_xlabel("Change in xwOBA")
ax.set_ylabel("Batters")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 17. Park Factors (Statcast, 2015+)
# New in 0.6.0: Savant's own park factors, 1-year and 3-year windows (100 = neutral). The FanGraphs table is still available as `park_factors_fangraphs`.
#
# The Athletics have played at Sutter Health Park (Sacramento) since 2025, so Savant has no 3-year factor for it yet; it is left out of the charts.

# %%
df_pf = fetch_years(sx.park_factors)
print(df_pf[df_pf.pf_3yr.isna()][["year", "team", "venue_name", "pf_1yr"]])
d = df_pf[df_pf.year == YEAR].dropna(subset=["pf_3yr", "pf_hr"]).sort_values("pf_3yr")
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].barh(d.team, d.pf_3yr, color=["#C44E52" if v > 100 else "#4C72B0" for v in d.pf_3yr])
axes[0].axvline(100, color="gray", ls="--")
axes[0].set_title(f"3-year park factor, runs ({YEAR})")
axes[0].set_xlabel("Park factor")
axes[1].scatter(d.pf_3yr, d.pf_hr, s=40, color="steelblue")
for _, r in d.iterrows():
    axes[1].annotate(r.team, (r.pf_3yr, r.pf_hr), fontsize=9, ha="center", va="bottom")
axes[1].axhline(100, color="gray", lw=0.8, ls="--")
axes[1].axvline(100, color="gray", lw=0.8, ls="--")
axes[1].set_title(f"Runs vs home runs ({YEAR})")
axes[1].set_xlabel("Park factor, runs")
axes[1].set_ylabel("Park factor, HR")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 18. Outs Above Average and 19. Outfield Jump
# Not wrapped by savant-extras; Savant's CSV endpoints work directly. OAA's own `year` column is empty, so it is set from the request.

# %%
frames = []
for y in YEARS:
    d = savant_csv("https://baseballsavant.mlb.com/leaderboard/outs_above_average"
                   f"?type=Fielder&startYear={y}&endYear={y}&split=no&team=&range=year&min=q&pos=&roles=&viz=hide&csv=true")
    d["year"] = y
    frames.append(d)
    time.sleep(1)
df_oaa = pd.concat(frames, ignore_index=True)
df_oj = savant_csv(f"https://baseballsavant.mlb.com/leaderboard/outfield_jump?year={YEAR}&min=q&csv=true")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
barh(axes[0], df_oaa[df_oaa.year == YEAR], "last_name, first_name", "outs_above_average",
     f"Outs above average, {YEAR}", "OAA")
axes[1].scatter(df_oj.rel_league_reaction_distance, df_oj.rel_league_routing_distance, s=30, alpha=0.6, color="darkorange")
axes[1].axhline(0, color="gray", lw=0.8, ls="--")
axes[1].axvline(0, color="gray", lw=0.8, ls="--")
axes[1].set_title(f"Outfielders: first step vs route, {YEAR}")
axes[1].set_xlabel("Reaction vs league (ft)")
axes[1].set_ylabel("Route vs league (ft)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 20. ABS Challenges (new in 0.6.0)
# MLB adopted the Automated Ball-Strike challenge in 2026; Triple-A has used it since 2025. One row per challenger, keyed by the MLBAM `player_id`.

# %%
rows = []
for level, year in [("aaa", 2025), ("aaa", 2026), ("mlb", 2026)]:
    for who in ["batter", "catcher"]:
        d = sx.abs_challenges(year, level=level, challenge_type=who)
        rows.append({"board": f"{level.upper()} {year}", "who": who,
                     "challenges": d.n_challenges.sum(), "won": d.n_overturns.sum()})
        time.sleep(1)
abs_tab = pd.DataFrame(rows)
abs_tab["share_won"] = abs_tab.won / abs_tab.challenges
print(abs_tab)

ax = abs_tab.pivot(index="board", columns="who", values="share_won").plot(kind="bar", rot=0, figsize=(9, 5.5))
ax.axhline(0.5, color="gray", ls="--")
ax.set_ylim(0, 0.75)
ax.legend(title="Challenger", loc="upper left", ncol=2)
ax.set_title("Share of ABS challenges won")
ax.set_xlabel("")
ax.set_ylabel("Share won")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 21. Triple-A pitch-level Statcast (new in 0.6.0)
# `statcast_minors` sends `minors=true`, which pybaseball's `statcast()` cannot, one request per day. Here one week of Triple-A.

# %%
df_aaa = sx.statcast_minors("2026-06-01", "2026-06-07")
print(f"{len(df_aaa):,} Triple-A pitches, {df_aaa.pitcher.nunique()} pitchers")
velo = df_aaa.dropna(subset=["pitch_type", "release_speed"]).groupby("pitch_type").release_speed.agg(["count", "mean"])
velo = velo[velo["count"] >= 200].sort_values("mean")
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(velo.index, velo["mean"], color="#4C72B0")
ax.set_xlim(70, None)
ax.set_title("Triple-A average velocity by pitch type (June 1-7, 2026)")
ax.set_xlabel("Release speed (mph)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## The published dataset
# The attached dataset holds these leaderboards for 2024-2026 as CSVs, built with the same package and checked (seasons present, no two seasons the same table, all pitch types). For example, velocity by pitch type across three seasons:

# %%
import glob, os
paths = glob.glob("/kaggle/input/**/pitch_movement.csv", recursive=True)
if paths:
    pm = pd.read_csv(paths[0])
    core = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]
    t = (pm[pm.pitch_type.isin(core)]
         .groupby(["pitch_type", "year"])
         .apply(lambda g: (g.avg_speed * g.pitches_thrown).sum() / g.pitches_thrown.sum())
         .unstack())
    print("Average speed (mph), weighted by pitches thrown")
    display(t.round(1))
    print("Files:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(os.path.dirname(paths[0]), "*.csv"))))
else:
    print("Attach the dataset yasunorim/baseball-savant-leaderboards-2024 to run this cell.")

# %% [markdown]
# ---
# ## Summary
#
# | Category | Functions |
# |---|---|
# | Batting | `bat_tracking`, `batted_ball`, `home_runs`, `swing_take`, `year_to_year` |
# | Pitching | `pitch_tempo`, `pitch_movement`, `pitcher_arm_angle`, `running_game`, `timer_infractions` |
# | Catching | `catcher_blocking`, `catcher_throwing`, `catcher_stance` |
# | Baserunning | `baserunning`, `basestealing` |
# | Fielding | `arm_strength` |
# | Park | `park_factors` (Statcast), `park_factors_fangraphs` |
# | ABS | `abs_challenges` (MLB 2026+, Triple-A 2025+) |
# | Minor leagues | `statcast_minors` (Triple-A pitch by pitch) |
#
# Most functions also have a `_range` version for several seasons.
#
# - **PyPI**: [savant-extras](https://pypi.org/project/savant-extras/)
# - **GitHub**: [yasumorishima/savant-extras](https://github.com/yasumorishima/savant-extras)
# - **Dataset**: [Baseball Savant Leaderboards (2024-2026)](https://www.kaggle.com/datasets/yasunorim/baseball-savant-leaderboards-2024)
# - **ABS dataset**: [ABS Challenges: Triple-A 2025 to MLB 2026](https://www.kaggle.com/datasets/yasunorim/mlb-abs-challenges-aaa-2025-to-mlb-2026)
#
# Data: Baseball Savant (MLB Advanced Media).
