# %% [markdown]
# # Ohtani 2026: Could the Slump Be Seen Early?
#
# Shohei Ohtani's bat cooled in 2026. After three seasons above a 1.000 OPS (2023-2025), he finished at .889 with 30 home runs, and on September 8 he went on the injured list with **right biceps inflammation**.
#
# Looking back, it is tempting to say the signs were there. This notebook asks the question the way it would have to be asked **during** the season: using only the pitches that had already happened on each day, would his own Statcast data have raised a flag before the injured list, and how often does the same flag go up in seasons when nothing was wrong?
#
# Data: **[Japanese MLB Players Statcast 2015-2026](https://www.kaggle.com/datasets/yasunorim/japan-mlb-pitchers-batters-statcast)** (every regular-season pitch Ohtani saw or threw, with bat tracking from 2024).
#
# **The answer, in short**
# - With a rule fixed before the 2026 series was looked at, his process data did flag in 2026, and sometimes a week or two before his results did (bat speed on April 26 before results on May 12; hard contact from July 7 before results on July 25). But process flags also came and went with no results flag after them (June; August 4-11 and 17-18), and the flag that was up on the day before the injured list was the results.
# - The same rule raises one or two flags a season in seasons with no injury. Ohtani 2026 sits at the top of that range; only hard contact is one flag above it.
# - As a pitcher, his four-seamer averaged 96.8-99.8 mph per start through his last start on July 3.
# - Not pre-registered: his bat speed fell 3.1 mph over the ten game days ending August 4, more than in any 10-day span of the reference seasons, and then came back.
#
# Contents
# 1. The data and the season
# 2. The rule, fixed in advance
# 3. 2026, day by day
# 4. How often the same rule fires when nothing is wrong
# 5. The pitching side: four-seam velocity
# 6. Not pre-registered: how fast bat speed fell
# 7. The 2026 season as it unfolded (GIF)
# 8. What this can and cannot say

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
                     "xtick.labelsize": 12, "ytick.labelsize": 12})

# On Kaggle the dataset is under /kaggle/input/; KAGGLE_INPUT_ROOT lets the notebook run elsewhere.
INPUT_ROOT = os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input")
paths = glob.glob(os.path.join(INPUT_ROOT, "**", "japanese_mlb_batting.csv"), recursive=True)
if not paths:
    raise FileNotFoundError("Attach the dataset yasunorim/japan-mlb-pitchers-batters-statcast to this notebook.")
DATA_DIR = os.path.dirname(paths[0])

BAT_COLS = ["batter", "player_name_eng", "game_year", "game_date", "game_pk", "at_bat_number", "pitch_number",
            "type", "description", "events", "launch_speed", "bat_speed", "woba_value", "woba_denom"]
bat = pd.read_csv(os.path.join(DATA_DIR, "japanese_mlb_batting.csv"), usecols=BAT_COLS, low_memory=False)
pit = pd.read_csv(os.path.join(DATA_DIR, "japanese_mlb_pitching.csv"), low_memory=False,
                  usecols=["pitcher", "game_year", "game_date", "game_pk", "at_bat_number", "pitch_number",
                           "pitch_type", "release_speed"])
summary = pd.read_csv(os.path.join(DATA_DIR, "season_summary.csv"))
for df in (bat, pit):
    df["game_date"] = pd.to_datetime(df["game_date"])
bat = bat.sort_values(["batter", "game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
pit = pit.sort_values(["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
OHTANI = 660271
print(f"{len(bat):,} pitches seen, {len(pit):,} pitches thrown")

# %% [markdown]
# ## 1. The data and the season
#
# wOBA rolls every way of reaching base into one number (league average is about .310-.320). Ohtani's 2026 was still far above average; it is the drop from his own 2023-2025 level that people noticed.

# %%
s = summary[(summary.mlbam_id == OHTANI) & (summary.role == "batter") & (summary.season >= 2021)]
s = s.set_index("season")
# Gate: plate appearances in the pitch file match MLB's count (within 1) for every season shown.
pa_file = bat[(bat.batter == OHTANI) & bat.events.notna()].groupby("game_year").size()
assert (pa_file.loc[s.index] - s.bat_pa).abs().max() <= 1
# Savant's per-pitch wOBA weights are not MLB's, so the per-PA wOBA used later reads a few points higher.
o = bat[(bat.batter == OHTANI) & bat.woba_denom.gt(0)]
mine = o.groupby("game_year").woba_value.sum() / o.groupby("game_year").woba_denom.sum()
print(pd.DataFrame({"PA": s.bat_pa.astype(int), "HR": s.bat_hr.astype(int), "OPS": s.bat_ops,
                    "wOBA (MLB)": s.bat_woba.round(3), "wOBA (Savant weights)": mine.loc[s.index].round(3)}).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#c0392b" if y == 2026 else "0.65" for y in s.index]
ax.bar(s.index.astype(str), s.bat_woba, color=colors)
for x, v in zip(s.index.astype(str), s.bat_woba):
    ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=12)
ax.set_ylim(0.25, 0.47)
ax.set_ylabel("wOBA")
ax.set_title("2026: down from three seasons above .415", fontsize=17, fontweight="bold")
ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. The rule, fixed in advance
#
# A flag is only useful if it would have been raised **at the time**. So the rule below was written down before any daily series (file `FREEZE.md` in the [GitHub folder](https://github.com/yasumorishima/kaggle-datasets/tree/main/ohtani-2026-early-warning), md5 `1005a90c99a5d55e55ff5dc7ac23a0a9`) in this notebook was computed. What had already been seen: the 2026 monthly totals, the injured-list date, and the date of his 2023 elbow injury.
#
# **Signals**, each computed on day *t* from the last *n* events **before** day *t*:
#
# | Signal | What it is | Window |
# |---|---|---|
# | EV90 | 90th percentile exit velocity (his best contact) | last 40 batted balls |
# | Hard-hit | share of batted balls at 95 mph or more | last 40 batted balls |
# | Bat speed | mean bat speed of swings at 50 mph or more (2024 on) | last 100 swings |
# | Whiff | misses per swing | last 150 swings |
# | wOBA | the **result**, for comparison | last 100 plate appearances |
# | Four-seam velocity | mean four-seamer speed | each start (30+ pitches) |
#
# **Line**: the 5th percentile of the same windowed signal over every day of the previous season (95th for whiff).
# **Flag**: worse than the line on 3 game days in a row. One flag = one run of such days.
#
# Event seasons: Ohtani 2026 (injured list from September 8) and 2023 (he left his August 23 start and the Angels announced a torn elbow ligament that night, [MLB.com](https://www.mlb.com/angels/news/what-s-next-for-shohei-ohtani-angels-after-2023-ucl-tear); his only 2023 injured-list placement was an oblique strain on September 16).
#
# Two details the frozen text did not spell out: a start is a game where he threw 30 or more pitches (counting every game moves the 2025 line from 97.24 to 97.27 mph and changes no flag), and flags are shown per season (the frozen text said per 100 team games; both are shown in section 4).
# Reference seasons: every hitter-season in the dataset with a previous season of 300+ plate appearances and **no** injured-list placement that year (MLB StatsAPI transactions).

# %%
SWING = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
         "missed_bunt", "bunt_foul_tip"}
MISS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
LOWER_IS_WORSE = {"EV90": True, "Hard-hit": True, "Bat speed": True, "Whiff": False, "wOBA": True}
UNITS = {"EV90": "mph, last 40 BIP", "Hard-hit": "share", "Bat speed": "mph, last 100 swings",
         "Whiff": "misses / swing", "wOBA": "last 100 PA"}
import matplotlib.dates as mdates


def shade(ax, index, up):
    """One day-wide band per game day with a flag up (no band over days he did not play)."""
    return [ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color="#e74c3c", alpha=0.25, lw=0)
            for d in index[up]]


def event_streams(p, y):
    """Per signal: (event dates, values, window, statistic) for player p in season y, plus his game days."""
    b = bat[(bat.batter == p) & (bat.game_year == y)]
    bip = b[(b.type == "X") & b.launch_speed.notna()]
    sw = b[b.description.isin(SWING)]
    bs = sw[sw.bat_speed >= 50]
    pa = b[b.woba_denom.gt(0)]
    return {
        "EV90": (bip.game_date.to_numpy(), bip.launch_speed.to_numpy(), 40, lambda v: np.percentile(v, 90)),
        "Hard-hit": (bip.game_date.to_numpy(), (bip.launch_speed >= 95).to_numpy(float), 40, np.mean),
        "Bat speed": (bs.game_date.to_numpy(), bs.bat_speed.to_numpy(), 100, np.mean),
        "Whiff": (sw.game_date.to_numpy(), sw.description.isin(MISS).to_numpy(float), 150, np.mean),
        "wOBA": (pa.game_date.to_numpy(), np.c_[pa.woba_value.to_numpy(), pa.woba_denom.to_numpy()], 100,
                 lambda v: v[:, 0].sum() / v[:, 1].sum()),
    }, np.unique(b.game_date.to_numpy())


def windowed(dates, vals, w, f, days):
    """Signal on each game day t from the last w events strictly before t (NaN until w exist)."""
    out = []
    for t in days:
        k = np.searchsorted(dates, t, side="left")
        out.append(f(vals[k - w:k]) if k >= w else np.nan)
    return np.array(out, float)


def flags(x, line, lower_worse, run=3):
    """Boolean 'flag up' per day (from the 3rd bad day of a run on), number of flags, first flag index."""
    bad = np.where(np.isnan(x), False, (x < line) if lower_worse else (x > line))
    up, streak, n, first = np.zeros(len(x), bool), 0, 0, None
    for i, z in enumerate(bad):
        streak = streak + 1 if z else 0
        if streak >= run:
            up[i] = True
        if streak == run:
            n += 1
            first = i if first is None else first
    return up, n, first


def run_season(p, y, prev):
    cur, days = event_streams(p, y)
    base, bdays = event_streams(p, prev)
    out = {}
    for k, (dt, v, w, f) in cur.items():
        bser = windowed(*base[k][:2], base[k][2], base[k][3], bdays)
        bser = bser[~np.isnan(bser)]
        if len(bser) < 20 or len(dt) < w:  # no bat tracking before 2024
            continue
        line = np.percentile(bser, 5 if LOWER_IS_WORSE[k] else 95)
        x = windowed(dt, v, w, f, days)
        up, n, first = flags(x, line, LOWER_IS_WORSE[k])
        out[k] = dict(line=line, x=pd.Series(x, index=pd.to_datetime(days)), up=up, n=n,
                      first=None if first is None else pd.Timestamp(days[first]))
    return out


# Planted check: the window must use only earlier days. Changing one 2026 batted ball's value to 130 mph
# date may change the signal on later days but never on days up to the moved one.
_dt, _v, _w, _f = event_streams(OHTANI, 2026)[0]["EV90"]
_days = np.unique(bat[(bat.batter == OHTANI) & (bat.game_year == 2026)].game_date.to_numpy())
_a = windowed(_dt, _v, _w, _f, _days)
_v2 = _v.copy(); _v2[200] = 130.0  # a planted 130 mph ball
_b = windowed(_dt, _v2, _w, _f, _days)
_k = np.searchsorted(_days, _dt[200], side="right")  # first game day after that ball
assert np.array_equal(_a[:_k], _b[:_k], equal_nan=True) and not np.array_equal(_a, _b, equal_nan=True)

# Injured-list placements of the hitters below (MLB StatsAPI transactions, fetched 2026-10-09).
IL_YEARS = {660271: {2018, 2019, 2023, 2026}, 673548: {2022, 2023, 2024, 2026}, 807799: {2024, 2025, 2026},
            663457: {2023, 2024, 2025, 2026}, 493114: {2015}, 400085: set()}
EVENTS = {2026: ("2026-09-08", "injured list (right biceps inflammation)"),
          2023: ("2023-08-23", "elbow ligament tear (left his start)")}

pa_by = bat[bat.events.notna()].groupby(["batter", "game_year"]).size()
REFS = [(p, y) for (p, y) in pa_by.index
        if pa_by.get((p, y - 1), 0) >= 300 and y not in IL_YEARS.get(p, {y}) and (p, y) != (OHTANI, 2021)]
assert all(p in IL_YEARS for p, _ in REFS)  # every reference hitter's injury history was checked
names = bat.drop_duplicates("batter").set_index("batter").player_name_eng
print("Reference seasons:", ", ".join(f"{names[p]} {y}" for p, y in REFS))

# %% [markdown]
# ## 3. 2026, day by day
#
# Grey line: the flag line from his 2025 season. Red shading: a flag is up. The dashed line is the injured list.

# %%
r26 = run_season(OHTANI, 2026, 2025)
IL26 = pd.Timestamp(EVENTS[2026][0])
LAST_START = pd.Timestamp("2026-07-03")
show = ["EV90", "Bat speed", "wOBA"]
fig, axes = plt.subplots(len(show), 1, figsize=(12, 10), sharex=True)
for ax, k in zip(axes, show):
    v = r26[k]
    ax.plot(v["x"].index, v["x"], color="#2c3e50", lw=2)
    ax.axhline(v["line"], color="0.5", lw=1.5)
    shade(ax, v["x"].index, v["up"])
    ax.axvline(IL26, color="#c0392b", ls="--", lw=1.5)
    ax.set_ylabel(f"{k}\n({UNITS[k]})")
axes[0].text(IL26, axes[0].get_ylim()[1], " injured list", color="#c0392b", va="top", fontsize=12)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
axes[-1].set_xlabel("2026 (red: flag up on that game day)")
fig.suptitle("August flags came and went; results flagged the day before the injured list", fontsize=17, fontweight="bold")
plt.tight_layout()
plt.show()

rows = []
for y, (d, what) in EVENTS.items():
    r = r26 if y == 2026 else run_season(OHTANI, y, y - 1)
    for k, v in r.items():
        pre = v["up"][v["x"].index < pd.Timestamp(d)]
        last3w = v["up"][(v["x"].index < pd.Timestamp(d)) & (v["x"].index >= pd.Timestamp(d) - pd.Timedelta(days=21))]
        rows.append({"season": y, "signal": k, "first flag": v["first"].date() if v["first"] is not None else "-",
                     "flags before event": int(np.sum(np.diff(np.r_[0, pre.astype(int)]) == 1)),
                     "flag up in last 3 weeks before": bool(last3w.any())})
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))

# The dates quoted below, checked against what the rule produced.
def up_days(r, k):
    return set(r[k]["x"].index[r[k]["up"]].strftime("%m-%d"))
r23 = run_season(OHTANI, 2023, 2022)
assert str(r26["EV90"]["first"].date()) == "2026-05-11" and str(r26["wOBA"]["first"].date()) == "2026-05-12"
assert str(r26["Bat speed"]["first"].date()) == "2026-04-26"
aug = lambda r, k: sorted(d for d in up_days(r, k) if "07-25" <= d <= "09-07")
assert aug(r26, "Bat speed") == ["08-04", "08-05", "08-07", "08-08", "08-09", "08-10", "08-11"]
assert aug(r26, "Hard-hit") == ["08-17", "08-18"] and aug(r26, "EV90") == []
assert sorted(d for d in up_days(r26, "wOBA") if "07-01" <= d <= "09-07") == ["07-25", "07-26", "07-28", "09-07"]
wk = r26["EV90"]["x"]["2026-09-01":"2026-09-07"]
assert len(wk) == 3
assert (wk == r26["EV90"]["line"]).all()  # exactly on the line, not below
def runs(r, k):
    idx, out = list(r[k]["x"].index), []
    for d in r[k]["x"].index[r[k]["up"]]:
        if out and idx.index(d) == idx.index(out[-1][1]) + 1:
            out[-1][1] = d
        else:
            out.append([d, d])
    return [f"{a:%m-%d}" if a == b else f"{a:%m-%d}..{b:%m-%d}" for a, b in out]


flag_runs = {k: runs(r26, k) for k in r26}
print(pd.Series({k: ", ".join(v) or "none" for k, v in flag_runs.items()}, name="2026 flag runs").to_string())
assert flag_runs == {"EV90": ["05-11..05-27", "06-05..06-07", "09-26..09-27"],
                     "Hard-hit": ["05-11..05-20", "07-07", "07-12..07-19", "08-17..08-18"],
                     "Bat speed": ["04-26", "06-13", "08-04..08-11", "09-25..09-27"], "Whiff": [],
                     "wOBA": ["05-12..05-18", "07-25..07-28", "09-07..09-27"]}
assert runs(r23, "Whiff") == ["05-31..06-17", "07-18..08-07"] and runs(r23, "wOBA") == []
ev = r26["EV90"]["x"]
assert ev["2026-08-30"] < r26["EV90"]["line"]
up_tie, _, _ = flags(np.where(ev.to_numpy() <= r26["EV90"]["line"], -1.0, 1.0), 0.0, True)
late_tie = ev.index[up_tie & (ev.index > pd.Timestamp("2026-08-19"))]
assert str(late_tie[0].date()) == "2026-09-02"  # first late-season flag if ties counted
summer23 = sorted(d for d in up_days(r23, "Whiff") if "07-01" <= d <= "08-23")
assert summer23[0] == "07-18" and summer23[-1] == "08-07" and r23["wOBA"]["n"] == 0

# %% [markdown]
# **2026.** Every flag run (printed above the chart):
#
# | Signal | Flag up (2026) |
# |---|---|
# | Bat speed | Apr 26; Jun 13; Aug 4-11; Sep 25-27 |
# | EV90 | May 11-27; Jun 5-7; Sep 26-27 |
# | Hard-hit | May 11-20; Jul 7, Jul 12-19; Aug 17-18 |
# | Whiff | none |
# | **wOBA (results)** | May 12-18; Jul 25-28; Sep 7-27 |
#
# Twice a process flag came before a results flag: bat speed on April 26 (results May 12) and hard contact from July 7 (results July 25). Three times process flags went up and cleared with no results flag following (June, August 4-11, August 17-18). Just before the injured list, EV90 dipped under its line on August 30 and then sat exactly on it on September 1, 2 and 7. Under the rule written in advance a tie is not a flag; had ties counted, EV90 would have flagged on September 2, two games before the results flag on September 7.
#
# **2023.** Batted-ball flags in late May, whiff flags May 31-June 17 and again July 18-August 7, and no results flag all season (he finished at a 1.066 OPS). The elbow injury came 16 days after the second whiff run ended. One event cannot tell whether that run meant anything.
#
# Whether any of this is a warning depends on how often the same flags go up when nothing is wrong.

# %% [markdown]
# ## 4. How often the same rule fires when nothing is wrong
#
# A flag that goes up every season is not a warning. Here is the same rule on the reference seasons, next to Ohtani 2026.

# %%
ref_rows = []
for p, y in REFS:
    for k, v in run_season(p, y, y - 1).items():
        ref_rows.append({"hitter": names[p], "season": y, "signal": k, "flags": v["n"]})
ref = pd.DataFrame(ref_rows)
print(ref.pivot_table(index=["hitter", "season"], columns="signal", values="flags").to_string())

per = ref.groupby("signal").flags.agg(["mean", "count"]).reindex(list(LOWER_IS_WORSE))
per["Ohtani 2026"] = [r26[k]["n"] if k in r26 else np.nan for k in per.index]
fig, ax = plt.subplots(figsize=(10, 5))
xx = np.arange(len(per))
ax.bar(xx - 0.2, per["mean"], 0.4, color="0.65", label="reference seasons (mean)")
ax.bar(xx + 0.2, per["Ohtani 2026"], 0.4, color="#c0392b", label="Ohtani 2026")
refmax_plot = ref.groupby("signal").flags.max().reindex(per.index)
ax.scatter(xx - 0.2, refmax_plot, marker="_", s=900, color="black", lw=2.5, label="highest reference season", zorder=3)
ax.set_xticks(xx, [f"{k}\n(n={int(c)})" for k, c in zip(per.index, per["count"])])
ax.set_ylabel("flags per season")
ax.set_ylim(0, 6.5)
ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=11)
ax.set_title("Healthy seasons raise the same flags one to two times", fontsize=17, fontweight="bold")
ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.show()
refmax = ref.groupby("signal").flags.max()
above = [k for k in per.index if per.loc[k, "Ohtani 2026"] > refmax[k]]
assert above == ["Hard-hit"] and per.loc["Hard-hit", "Ohtani 2026"] == 4 and refmax["Hard-hit"] == 3
assert refmax["Bat speed"] == 5 and len(REFS) == 9
games = {2020: 60}
ref["per100"] = ref["flags"] / ref.season.map(lambda y: games.get(y, 162)) * 100
o100 = {k: r26[k]["n"] / 162 * 100 for k in r26}
above100 = [k for k in o100 if o100[k] > ref[ref.signal == k].per100.max()]
print("flags per 100 team games, reference max:", ref.groupby("signal").per100.max().round(2).to_dict())
print("Ohtani 2026:", {k: round(v, 2) for k, v in o100.items()})
assert above100 == ["Hard-hit"]
pre = {k: flags(r26[k]["x"][r26[k]["x"].index < IL26].to_numpy(), r26[k]["line"], LOWER_IS_WORSE[k])[1] for k in r26}
assert pre == {"EV90": 2, "Hard-hit": 4, "Bat speed": 3, "Whiff": 0, "wOBA": 3}, pre

# %% [markdown]
# Ohtani 2026 is at the top of the reference range: at or below the highest healthy season on every signal except hard contact (4 flags against at most 3), and Ohtani 2024 had 5 bat-speed flags with no injury. The 2026 counts include his five games after the injured list; before it they were EV90 2, hard-hit 4, bat speed 3, wOBA 3. Per 100 team games (the measure the frozen text named; Ohtani 2020 was a 60-game season) the picture is the same: hard contact is the only signal above every reference season. Part of this is the baseline itself: a line drawn from a career season is easy to fall under. With nine reference seasons, one signal one flag above the range is not enough to tell 2026 apart from an ordinary slump.

# %% [markdown]
# ## 5. The pitching side: four-seam velocity
#
# The injury was to his right (throwing) arm, so the first place to look might be his fastball. Each dot is one start.

# %%
def starts(y):
    g = pit[(pit.pitcher == OHTANI) & (pit.game_year == y)]
    n = g.groupby("game_date").size()
    ff = g[g.pitch_type == "FF"].groupby("game_date").release_speed.mean()
    return ff[ff.index.isin(n[n >= 30].index)].dropna()


s25, s26 = starts(2025), starts(2026)
line_ff = np.percentile(s25, 5)
up_ff, n_ff, _ = flags(s26.to_numpy(), line_ff, True)
print(f"2025 starts: {len(s25)}, 2026 starts: {len(s26)} (last {s26.index.max().date()}), flags: {n_ff}")
assert s26.index.max() == LAST_START
below = s26[s26 < line_ff]
assert list(below.index) == [pd.Timestamp("2026-03-31")] and n_ff == 0
s22, s23 = starts(2022), starts(2023)
line23 = np.percentile(s22, 5)
b23 = s23[s23 < line23]
n23 = flags(s23.to_numpy(), line23, True)[1]
print(f"2023: line {line23:.2f}, starts under it {[d.strftime('%m-%d') for d in b23.index]}, flags {n23}")
assert [d.strftime("%m-%d") for d in b23.index] == ["06-09", "07-04", "08-09"] and n23 == 0
n_aug23 = int((pit.pitcher.eq(OHTANI) & pit.game_date.eq(pd.Timestamp("2023-08-23"))).sum())
print("pitches thrown on 2023-08-23:", n_aug23)
assert 0 < n_aug23 < 30

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(s26.index, s26, "o-", color="#2c3e50", lw=2, ms=8)
ax.axhline(line_ff, color="0.5", lw=1.5)
ax.text(pd.Timestamp("2026-07-20"), line_ff + 0.05, "flag line (2025 starts, 5th pct)", va="bottom", fontsize=12, color="0.4")
ax.axvline(IL26, color="#c0392b", ls="--", lw=1.5)
ax.text(IL26, ax.get_ylim()[1], " injured list", color="#c0392b", va="top", ha="right", fontsize=12)
ax.set_xlim(s26.index.min() - pd.Timedelta(days=7), IL26 + pd.Timedelta(days=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_ylabel("Four-seamer, mean mph")
ax.set_title("His fastball held up through his last start on July 3", fontsize=17, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# His first start (March 31, 96.8 mph) was under the line; every start after it was above, so the rule never flagged. He made no start after July 3 and was not on the injured list then; this data does not say why.
#
# 2023, against the 2022 line (95.55 mph): three starts were under it (June 9, July 4, August 9), never three in a row, so no flag. His August 23 start, cut short by the injury, is under 30 pitches and not counted as a start here.

# %% [markdown]
# ## 6. Not pre-registered: how fast bat speed fell
#
# The rule above looks at the **level** of a signal. After seeing the results, a natural next question is the **speed** of a drop. This section was not fixed in advance, so read it as a lead, not a finding.
#
# For every game day: bat speed (last 100 swings) minus its value 10 game days earlier. Reference: the seasons with bat tracking and no injured list.

# %%
def bat_speed(p, y):
    cur, days = event_streams(p, y)
    return pd.Series(windowed(*cur["Bat speed"][:2], 100, np.mean, days), index=pd.to_datetime(days)).dropna()


ref_bs = [(p, y) for p, y in REFS if y >= 2024]
drops = pd.concat([(bat_speed(p, y) - bat_speed(p, y).shift(10)).dropna() for p, y in ref_bs])
o26 = bat_speed(OHTANI, 2026)
d26 = (o26 - o26.shift(10)).dropna()
pre_il = d26[d26.index < IL26].iloc[-1]
print("Reference:", ", ".join(f"{names[p]} {y}" for p, y in ref_bs), f"({len(drops)} days)")
print(f"Largest 2026 drop: {d26.min():.2f} mph on {d26.idxmin().date()}; reference minimum {drops.min():.2f}")
print(f"Last day before the injured list: {pre_il:.2f} mph; {np.mean(drops <= pre_il):.1%} of reference days fell at least as much")
_bk = o26["2026-08-21":"2026-08-25"]; assert len(_bk) == 4 and round(_bk.min(), 1) == 74.4 and round(_bk.max(), 1) == 75.1
assert d26.idxmin() == pd.Timestamp("2026-08-04") and d26.min() < drops.min() and 0.02 < np.mean(drops <= pre_il) < 0.045

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(drops, bins=40, color="0.7")
for v, lab in [(d26.min(), f"Aug 4: {d26.min():.1f}"), (pre_il, f"Sep 7: {pre_il:.1f}")]:
    ax.axvline(v, color="#c0392b", lw=2)
    ax.text(v, ax.get_ylim()[1] * 0.95, f" {lab}", color="#c0392b", fontsize=12, va="top")
ax.set_xlabel("Change in bat speed over 10 game days (mph)")
ax.set_ylabel("days")
ax.set_title("The early-August drop was bigger than any healthy-season day", fontsize=17, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# The steepest fall of 2026 ended on August 4 (the ten game days from late July), a month before the injured list, and it was larger than any 10-day change in the reference seasons. But his bat speed was back at 74.4-75.1 mph from August 21 to 25, and the drop just before the injured list was ordinary (about 1 in 30 healthy days fall as much). With three reference seasons, one unusual drop that then reversed is not enough to call it a warning.

# %% [markdown]
# ## 7. The 2026 season as it unfolded
#
# The same three signals, drawn day by day as they would have looked at the time: every point and every red band uses only earlier days. (The axes are fixed for the whole season so the picture does not jump.)

# %%
from IPython.display import Image, display
from matplotlib.animation import FuncAnimation, PillowWriter

days = r26["EV90"]["x"].index
frames = list(range(len(days))) + [len(days) - 1] * 18  # hold the last day
fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
fig.subplots_adjust(left=0.13, right=0.97, top=0.9, bottom=0.07, hspace=0.12)  # fixed, so nothing jumps
fig.suptitle("Ohtani 2026: three signals, one day at a time", fontsize=17, fontweight="bold")
lines, shades = [], []
for ax, k in zip(axes, show):
    v = r26[k]
    lo, hi = np.nanmin(v["x"]), np.nanmax(v["x"])
    pad = (hi - lo) * 0.15
    ax.set_ylim(min(lo, v["line"]) - pad, max(hi, v["line"]) + pad)  # fixed from the start
    ax.axhline(v["line"], color="0.5", lw=1.5)
    ax.set_ylabel(f"{k}\n({UNITS[k]})")
    lines.append(ax.plot([], [], color="#2c3e50", lw=2.2)[0])
axes[-1].set_xlim(days.min() - pd.Timedelta(days=3), days.max() + pd.Timedelta(days=3))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
il_text = axes[0].text(IL26, axes[0].get_ylim()[1], " injured list", color="#c0392b", va="top", fontsize=12,
                       visible=False)
date_text = axes[0].text(0.01, 0.95, "", transform=axes[0].transAxes, fontsize=22, fontweight="bold",
                         color="0.35", va="top")
il_lines = [ax.axvline(IL26, color="#c0392b", ls="--", lw=1.5, visible=False) for ax in axes]


def draw(i):
    for c in shades:
        c.remove()
    shades.clear()
    t = days[i]
    for ax, ln, k in zip(axes, lines, show):
        x = r26[k]["x"]
        ln.set_data(x.index[: i + 1], x.to_numpy()[: i + 1])
        up = r26[k]["up"][: i + 1]
        shades.extend(shade(ax, x.index[: i + 1], up))
    for il in [*il_lines, il_text]:
        il.set_visible(t >= IL26)
    date_text.set_text(t.strftime("%b %d"))
    # Gate on what is drawn: each line is exactly the signal up to day t, and no later day.
    for ln, k in zip(lines, show):
        xs, ys = ln.get_data()
        assert len(xs) == i + 1 and pd.Timestamp(xs[-1]) == t
        assert np.array_equal(ys, r26[k]["x"].to_numpy()[: i + 1], equal_nan=True)
    assert all(il.get_visible() == (t >= IL26) for il in [*il_lines, il_text])
    assert len(shades) == sum(int(r26[k]["up"][: i + 1].sum()) for k in show)  # one band per flagged day so far
    return [*lines, *shades, date_text, *il_lines, il_text]


for i in range(len(days)):  # run every frame's gate before writing anything
    draw(i)
gif_path = "ohtani_2026_signals.gif"
FuncAnimation(fig, draw, frames=frames, blit=False).save(gif_path, writer=PillowWriter(fps=10))
plt.close(fig)
from PIL import Image as PILImage

with PILImage.open(gif_path) as im:
    ms = 0
    for k in range(im.n_frames):
        im.seek(k)
        ms += im.info.get("duration", 0)
print(f"{len(frames)} frames, {ms / 1000:.1f} s, {os.path.getsize(gif_path) / 1024:.0f} KB")
display(Image(filename=gif_path))

# %% [markdown]
# ## 8. What this can and cannot say
#
# - **Can say:** with this public pitch-by-pitch data and a rule fixed before looking, Ohtani's process data did flag in 2026, sometimes a week or two before his results, but it also flagged and cleared with nothing following, and on the day before the injured list the flag that was up was the results.
# - **Can say:** the same rule raises one or two flags in a season with no injury, so a flag on its own would have meant little.
# - **Cannot say:** whether his condition was already worse. Teams have data this does not: workload, biomechanics, the player's own report. A biceps problem may change nothing that Statcast measures until it stops a player.
# - **Lead for later:** the speed of a bat-speed drop (section 6) is worth fixing as a rule now and testing on seasons that have not happened yet.
#
# Sources: Baseball Savant Statcast search and MLB StatsAPI (via the dataset); injured-list dates from MLB StatsAPI transactions.
