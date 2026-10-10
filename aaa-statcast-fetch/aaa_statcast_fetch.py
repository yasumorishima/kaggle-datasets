# %% [markdown]
# # Triple-A Statcast 2023-2026: pitch-by-pitch fetch with completeness gates
#
# Input for the pre-registered Triple-A replication of [Do Hitters Drift From Normal Before the IL?](https://www.kaggle.com/code/yasunorim/do-hitters-drift-from-normal-before-the-il). This notebook only fetches and checks pitches; it reads no injured-list data.
#
# - Source: Baseball Savant's minors search through [savant-extras](https://pypi.org/project/savant-extras/) `statcast_minors(level="AAA")`, one request a day.
# - Days: every day on which MLB StatsAPI lists a finished Triple-A regular-season game.
# - Gates (the notebook fails instead of writing a partial file): every such day returns CSV after retries, at least 97% of finished games are present in every season, no major-league game, regular season only.

# %%
import subprocess
import sys

import os

if os.environ.get("SKIP_PIP") != "1":  # set only for a local code test with the library on the path
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "savant-extras>=0.6.0"], check=True)

import hashlib
import os
import time
import warnings

import pandas as pd
import requests
import savant_extras
from savant_extras import statcast_minors

print("savant-extras", savant_extras.__version__)
SEASONS = [int(y) for y in os.environ.get("AAA_SEASONS", "2023,2024,2025,2026").split(",")]
MAX_DAYS = int(os.environ.get("AAA_MAX_DAYS", "0"))  # code test only: first N game days of each season
API = "https://statsapi.mlb.com/api/v1"
UA = {"User-Agent": "kaggle-notebook-research/1.0"}
KEEP = ["batter", "game_date", "game_pk", "at_bat_number", "pitch_number", "type", "description", "events",
        "zone", "stand", "hc_x", "hc_y", "launch_speed", "launch_angle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "home_team", "away_team", "game_type"]


def get(path, **params):
    for k in range(4):
        try:
            r = requests.get(f"{API}{path}", params=params, headers=UA, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"StatsAPI {path} failed")


# %% [markdown]
# ## Finished Triple-A regular-season games (StatsAPI schedule)

# %%
sched = []
for y in SEASONS:
    js = get("/schedule", sportId=11, season=y, gameType="R")
    for d in js["dates"]:
        for g in d["games"]:
            st = g["status"]
            sched.append({"season": y, "date": d["date"], "game_pk": g["gamePk"],
                          "abstract": st.get("abstractGameState"), "detailed": st.get("detailedState")})
sched = pd.DataFrame(sched)
print(sched.groupby(["season", "detailed"]).size().unstack(fill_value=0).to_string())
# A game is finished when it was played to an end on that date (postponed, cancelled and suspended are not).
done = sched[(sched.abstract == "Final") & sched.detailed.isin(["Final", "Completed Early", "Game Over"])]
done = done.drop_duplicates("game_pk")

# %% [markdown]
# ## Fetch, one request a day, with retries

# %%
missing_days, per_day = [], []
for y in SEASONS:
    days = sorted(done.loc[done.season == y, "date"].unique())
    if MAX_DAYS:
        days = days[:MAX_DAYS]
    frames = []
    for day in days:
        df = None
        last = ""
        for attempt in range(5):
            err = None
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    out = statcast_minors(day, day, level="AAA", player_type="batter", sleep=0)
                except (requests.RequestException, ValueError) as e:
                    out, err = None, repr(e)
            msgs = [str(x.message) for x in w]
            # A day with a finished game must return rows; an error page or an empty answer is retried.
            if out is not None and len(out) and not any("instead of CSV" in m for m in msgs):
                df = out
                break
            last = (err or "; ".join(msgs) or "no rows")[:200]
            time.sleep(10 * (attempt + 1))
        if df is None:
            missing_days.append((day, last))
            continue
        per_day.append({"season": y, "date": day, "rows": len(df), "games": df.game_pk.nunique() if len(df) else 0})
        if len(df):
            frames.append(df[[c for c in KEEP if c in df.columns]])
        time.sleep(1.0)
    pitches = pd.concat(frames, ignore_index=True)
    assert set(KEEP) <= set(pitches.columns), set(KEEP) - set(pitches.columns)
    pitches = pitches[KEEP]
    pitches["game_date"] = pd.to_datetime(pitches["game_date"])
    for c in ("batter", "game_pk", "at_bat_number", "pitch_number"):
        pitches[c] = pitches[c].astype("int64")
    pitches.to_parquet(f"aaa_statcast_{y}.parquet", index=False)
    print(f"{y}: {len(days)} game days, {len(pitches):,} pitches, {pitches.game_pk.nunique():,} games, "
          f"{pitches.batter.nunique():,} batters")

per_day = pd.DataFrame(per_day)
per_day.to_csv("aaa_per_day.csv", index=False)

# %% [markdown]
# ## Gates

# %%
assert not missing_days, f"days without CSV after retries: {missing_days}"
cover = []
for y in SEASONS:
    p = pd.read_parquet(f"aaa_statcast_{y}.parquet", columns=["game_pk", "game_type", "game_date"])
    want = done[done.season == y]
    if MAX_DAYS:
        want = want[want.date.isin(per_day.loc[per_day.season == y, "date"])]
    have = set(p.game_pk.unique())
    share = want.game_pk.isin(have).mean()
    extra = len(have - set(sched.game_pk))
    cover.append({"season": y, "finished games": len(want), "present": int(want.game_pk.isin(have).sum()),
                  "share": round(share, 4), "games not in the schedule": extra,
                  "game types": ",".join(sorted(p.game_type.unique())),
                  "first": p.game_date.min().date(), "last": p.game_date.max().date()})
cover = pd.DataFrame(cover)
print(cover.to_string(index=False))
assert (cover.share >= 0.97).all(), "a season is missing more than 3% of finished games"
assert (cover["game types"] == "R").all(), "non-regular-season games present"
want_missing = done[~done.game_pk.isin(set().union(*[set(pd.read_parquet(f"aaa_statcast_{y}.parquet", columns=["game_pk"]).game_pk) for y in SEASONS]))]
want_missing.to_csv("aaa_missing_games.csv", index=False)
cover.to_csv("aaa_coverage.csv", index=False)
for y in SEASONS:
    print(f"aaa_statcast_{y}.parquet md5 {hashlib.md5(open(f'aaa_statcast_{y}.parquet', 'rb').read()).hexdigest()}")
