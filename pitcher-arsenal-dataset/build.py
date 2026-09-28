"""Build the Kaggle dataset "MLB Pitcher Arsenal 2020-2026" (slug mlb-pitcher-arsenal-2020-2025).

Run by `.github/workflows/update-dataset.yml` (via build.sh) before `kaggle datasets version`.

Why this exists
- The previous version was a single wide CSV built by hand in a notebook from raw pitches. Its
  whiff_rate divided swinging strikes by swings *plus called strikes* (2025 four-seam median 0.132
  against Savant's 21.6%), it covered March-November (spring training and postseason included),
  kept PO (pitchouts) and unknown types as pitch types, and had a row with no name.
- This build takes every number from Baseball Savant's own regular-season leaderboards instead.

Sources (one season per request; Savant ignores parameters it does not read and then answers with
the current season, so the season of every answer is checked, see _fetch_html):
- Pitch Arsenal Stats (leaderboard/pitch-arsenal-stats, the page's embedded leaderboardData):
  one row per pitcher and pitch type: pitches, usage, run value, whiff%, K%, put-away%, wOBA/xwOBA.
  `min=1` (minimum PA ending on that pitch) and `minPitches=1` are Savant's lowest settings.
- Pitch Arsenals (leaderboard/pitch-arsenals, the page's embedded data): per pitcher and pitch
  type average speed, spin, break and release point. `min=1` is accepted; the page lists pitchers
  with at least 10 pitches and leaves out position players who pitched. Pitch types thrown but
  never ending a plate appearance are only here.
- Pitch Movement (savant_extras.pitch_movement, one request per pitch type): break compared with
  pitches of similar speed and release, for the pitchers Savant lists there.
- Pitcher Arm Angle (savant_extras.pitcher_arm_angle): the pitchers Savant lists there.

Every season 2020-2026 is fetched; the build fails and writes nothing unless every gate in
`gates()` passes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import savant_extras as sx

YEARS = tuple(range(2020, 2027))
SLEEP = 1.0
UA = {"User-Agent": "kaggle-datasets-build/1.0"}
# Pitch types Savant's arsenal leaderboards serve (knuckle curves are counted as CU there; eephus,
# pitchouts and unclassified pitches are not listed).
PITCH_TYPES = ("FF", "SI", "FC", "SL", "ST", "SV", "CU", "CH", "FS", "FO", "SC", "KN")
# Present in every season 2020-2026. Savant labels sweepers (ST) back to 2020 (56 pitchers in 2020,
# 383 in 2026 on the 2026-09-28 build), so ST is required too.
CORE_PITCH_TYPES = {"FF", "SI", "FC", "SL", "ST", "CU", "CH"}
OLD_FILE = "pitcher_arsenal_evolution_2020_2025.csv"
# Files of the previous version that this build replaces on purpose: the old CSV, and the two
# notebooks and README that computed and described it (they stay in the GitHub repository,
# dataset3_pitcher_arsenal/). Any other file in the current version fails the build.
OUTPUTS = ("pitcher_arsenal", "pitcher_arsenal_wide", "arsenal_changes")
REPLACED = {OLD_FILE, "pitcher_arsenal_evolution_2020_2025.ipynb", "pitcher_arsenal_analysis.ipynb", "README.md",
            *(f"{n}.csv" for n in OUTPUTS)}

STATS_URL = ("https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
             "?type=pitcher&pitchType=&year={y}&team=&min=1&minPitches=1")
ARSENAL_URL = "https://baseballsavant.mlb.com/leaderboard/pitch-arsenals?year={y}&min=1&type=avg_speed&hand="

# Gate thresholds, set from the 2020-2026 build of 2026-09-28 (the measured values are printed).
MIN_ROWS_LONG = 2000        # rows per season; measured 2,912 (2020, the 60-game season) to 3,896
MIN_PITCHERS = 500          # pitchers per season; measured 720 (2020) to 871
MIN_CHANGES = 2000          # year-over-year rows per season; measured 2,676 to 3,048
USAGE_MIN_PITCHES = 100     # the usage-sum gate looks at pitchers with at least this many pitches
USAGE_LO, USAGE_HI = 97.0, 100.6
# Share of those pitcher-seasons allowed outside [USAGE_LO, USAGE_HI]. Measured 0.10% (5 of 4,895):
# position players and Zack Greinke 2020, whose eephus (EP) Savant's arsenal pages do not list.
USAGE_MAX_BAD = 0.005
# Median whiff% of pitch types thrown 100+ times, per season. Measured FF 21.1-22.0, SI 13.0-15.0,
# FC 21.2-24.6, SL 32.4-36.8, CU 29.5-33.0, CH 28.8-32.4. The old file's FF median was 13.2.
WHIFF_MEDIAN = {"FF": (17, 27), "SI": (9, 19), "FC": (17, 29), "SL": (27, 41), "CU": (24, 38), "CH": (24, 37)}
COVER_ARSENAL = 0.98        # stats rows with speed/spin/break; measured 0.995-0.997
COVER_MOVEMENT = 0.40       # stats rows with the vs-similar break; measured 0.516-0.646
COVER_MOVEMENT_250 = 0.95   # the same for pitch types thrown 250+ times; measured 0.984-0.996
COVER_ARM = 0.20            # pitchers with an arm angle; measured 0.256 (2020) to 0.347
OLD_COVER = 0.99           # old version's pitcher-seasons found here; measured 0.998 (9 of 4,253 missing)
NEAR_SHARED, NEAR_EQUAL = 0.9, 0.9


# ----------------------------------------------------------------------------- fetch

def _fetch_html(url: str, var: str, year: int) -> list[dict]:
    time.sleep(SLEEP)
    r = requests.get(url, timeout=120, headers=UA)
    r.raise_for_status()
    h = r.text
    m = re.search(r"var " + var + r" = (\[.*?\]);\s*\n", h, re.S)
    if not m:
        raise SystemExit(f"{url}: no `var {var}` in the page")
    rows = json.loads(m.group(1))
    qs = re.search(r"var queryString = (\{.*?\});", h)
    if qs and str(json.loads(qs.group(1)).get("year")) != str(year):
        raise SystemExit(f"{url}: Savant read year {json.loads(qs.group(1)).get('year')}, not {year}")
    got = {str(r.get("year")) for r in rows}
    if got != {str(year)}:
        raise SystemExit(f"{url}: rows carry years {sorted(got)}, asked for {year}")
    return rows


def fetch_stats(y: int) -> pd.DataFrame:
    return pd.DataFrame(_fetch_html(STATS_URL.format(y=y), "leaderboardData", y))


def fetch_arsenal(y: int) -> pd.DataFrame:
    """Wide page data (one row per pitcher) melted to one row per pitcher and pitch type."""
    d = pd.DataFrame(_fetch_html(ARSENAL_URL.format(y=y), "data", y))
    frames = []
    for pt in PITCH_TYPES:
        p = pt.lower()
        cols = {f"{p}_avg_speed": "avg_speed", f"{p}_avg_spin": "avg_spin", f"{p}_avg_break_x": "break_x",
                f"{p}_avg_break_z": "break_z", f"{p}_avg_break_z_induced": "break_z_induced",
                f"{p}_rel_x": "release_x", f"{p}_rel_z": "release_z", f"n_{p}": "arsenal_usage"}
        have = [c for c in cols if c in d.columns]
        if f"{p}_avg_speed" not in have:
            continue
        f = d[["pitcher", "year", "pitch_hand"] + have].rename(columns=cols)
        f = f[pd.to_numeric(f["avg_speed"], errors="coerce").notna()].copy()
        f["pitch_type"] = pt
        frames.append(f)
    out = pd.concat(frames, ignore_index=True).rename(columns={"pitcher": "pitcher_id"})
    for c in out.columns:
        if c not in ("pitch_type", "pitch_hand"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def fetch_movement(y: int) -> pd.DataFrame:
    frames = []
    for pt in PITCH_TYPES:
        time.sleep(SLEEP)
        df = sx.pitch_movement(y, pitch_type=pt)  # checks the season and the pitch type itself
        if df.empty:
            continue
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if set(df["year"].astype(int)) != {y}:
        raise SystemExit(f"pitch_movement {y}: years {sorted(set(df['year']))}")
    return df


def fetch_arm_angle(y: int) -> pd.DataFrame:
    time.sleep(SLEEP)
    df = sx.pitcher_arm_angle(y)  # no season column: checked by the distinct-season gate
    df["year"] = y
    return df


FETCHERS = {"stats": fetch_stats, "arsenal": fetch_arsenal, "movement": fetch_movement, "arm_angle": fetch_arm_angle}


def fetch_all(cache: Path | None) -> dict[str, pd.DataFrame]:
    """cache is for development runs and the stale-season test only (the workflow never passes it)."""
    if cache and (cache / "raw.pkl").exists():
        print(f"using cached raw tables {cache / 'raw.pkl'}")
        return pickle.loads((cache / "raw.pkl").read_bytes())
    raw = {}
    for name, fn in FETCHERS.items():
        parts = []
        for y in YEARS:
            parts.append(fn(y))
            print(f"fetched {name} {y}: {len(parts[-1])} rows", flush=True)
        raw[name] = pd.concat(parts, ignore_index=True)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "raw.pkl").write_bytes(pickle.dumps(raw))
    return raw


# ----------------------------------------------------------------------------- tables

STAT_COLS = {  # Savant column -> output column
    "pitches": "pitches", "total_pitches": "total_pitches", "pitch_usage": "usage_pct", "pa": "pa",
    "run_value": "run_value", "run_value_unformatted": "run_value_exact", "run_value_per_100": "run_value_per_100",
    "whiff_percent": "whiff_pct", "k_percent": "k_pct", "put_away": "put_away_pct",
    "ba": "ba", "est_ba": "xba", "slg": "slg", "est_slg": "xslg", "woba": "woba", "est_woba": "xwoba",
    "hard_hit_percent": "hard_hit_pct", "launch_angle_avg": "launch_angle_avg",
    "exit_velocity_avg": "exit_velocity_avg", "barrel_batted_rate": "barrel_pct",
}
ARSENAL_COLS = ("avg_speed", "avg_spin", "break_x", "break_z", "break_z_induced", "release_x", "release_z")
# Savant's Pitch Movement leaderboard compares each pitch with pitches of similar speed and
# release. Its own break columns repeat Pitch Arsenals' with other sign conventions, so only the
# comparison is kept (measured on this build: diff_x = |break_x| - |similar break_x| in 97.6% of
# rows; diff_z is signed in the pitch's own direction: more rise for FF/FC, more drop otherwise).
MOVEMENT_COLS = {
    "diff_x": "break_x_vs_similar", "diff_z": "break_z_vs_similar",
    "percent_rank_diff_x": "break_x_vs_similar_pctile", "percent_rank_diff_z": "break_z_vs_similar_pctile",
}
KEY = ["pitcher_id", "season", "pitch_type"]


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    order = {pt: i for i, pt in enumerate(PITCH_TYPES)}
    return df.sort_values(["season", "pitcher_id", "pitch_type"],
                          key=lambda c: c.map(order) if c.name == "pitch_type" else c).reset_index(drop=True)


def build_long(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    s = raw["stats"]
    s = s[s["pitch_type"].isin(PITCH_TYPES)]
    long = pd.DataFrame({
        "pitcher_id": s["player_id"].astype(int), "player_name": s["player_name"].str.strip(),
        "season": s["year"].astype(int), "team": s["team_name_alt"],
        "pitch_type": s["pitch_type"], "pitch_name": s["pitch_name"],
    })
    for src, dst in STAT_COLS.items():
        long[dst] = pd.to_numeric(s[src], errors="coerce")

    ars = raw["arsenal"].rename(columns={"year": "season"})[[*KEY, *ARSENAL_COLS, "arsenal_usage"]]
    # A switch-pitcher (Pat Venditte, 2020) has one Pitch Arsenals row per hand; those cannot be
    # matched to the stats row (one per pitch type), so they are left empty.
    two = ars.duplicated(KEY, keep=False)
    if two.any():
        print(f"arsenal: {int(two.sum())} rows of pitchers with one row per hand left unmatched: "
              f"{sorted(set(map(tuple, ars.loc[two, ['pitcher_id', 'season']].to_numpy().tolist())))}")
    ars = ars[~two]
    long["in_arsenal_stats"] = True
    long = long.merge(ars, on=KEY, how="outer", validate="1:1", indicator=True)
    # Pitch types in Pitch Arsenals with no Pitch Arsenal Stats row: no plate appearance ended on
    # that pitch (the stats page lists a pitch from 1 PA). They keep usage, speed, spin and break
    # from Pitch Arsenals; pitches and the outcome columns stay empty.
    only = long["_merge"] == "right_only"
    per_pitcher = long.loc[~only].groupby(["pitcher_id", "season"]).agg(
        player_name=("player_name", "first"), team=("team", "first"), total_pitches=("total_pitches", "max"))
    known = long.loc[only, ["pitcher_id", "season"]].merge(per_pitcher, left_on=["pitcher_id", "season"],
                                                           right_index=True, how="left")
    orphan = pd.Series(False, index=long.index)
    orphan[only] = known["player_name"].isna().to_numpy()
    print(f"arsenal-only pitch types added: {int((only & ~orphan).sum())}; "
          f"dropped (pitcher-season not in Pitch Arsenal Stats at all): {int(orphan.sum())}")
    for c in ("player_name", "team", "total_pitches"):
        long.loc[only, c] = known[c].to_numpy()
    names = s.drop_duplicates("pitch_type").set_index("pitch_type")["pitch_name"]
    long.loc[only, "pitch_name"] = long.loc[only, "pitch_type"].map(names)
    long.loc[only, "usage_pct"] = long.loc[only, "arsenal_usage"]
    long.loc[only, "in_arsenal_stats"] = False
    long = long[~orphan].drop(columns=["_merge", "arsenal_usage"])
    long["in_arsenal_stats"] = long["in_arsenal_stats"].astype(bool)

    m = raw["movement"]
    mov = m[["pitcher_id", "year", "pitch_type", *MOVEMENT_COLS]].rename(columns={"year": "season", **MOVEMENT_COLS})
    mov["pitcher_id"] = mov["pitcher_id"].astype(int)
    mov["season"] = mov["season"].astype(int)
    long = long.merge(mov, on=KEY, how="left", validate="1:1")
    for c in ("pitches", "total_pitches", "pa", "run_value", "avg_spin"):
        long[c] = long[c].round().astype("Int64")
    long["run_value_exact"] = long["run_value_exact"].round(3)
    for c in ("break_x_vs_similar_pctile", "break_z_vs_similar_pctile"):
        long[c] = long[c].round(4)
    return _sort(long)


WIDE_METRICS = ("pitches", "usage_pct", "avg_speed", "avg_spin", "break_x", "break_z_induced",
                "whiff_pct", "put_away_pct", "run_value_per_100", "xwoba")


def build_wide(long: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    key = ["pitcher_id", "season"]
    base = (long.sort_values("usage_pct", ascending=False)
            .groupby(key, as_index=False)
            .agg(player_name=("player_name", "first"), team=("team", "first"), total_pitches=("total_pitches", "max"),
                 n_pitch_types=("pitch_type", "size"), usage_pct_sum=("usage_pct", "sum")))
    base["usage_pct_sum"] = base["usage_pct_sum"].round(1)
    # Throwing hand from Pitch Arsenals (not in the stats table); S = a row for each hand.
    ah = raw["arsenal"].rename(columns={"year": "season"}).groupby(key)["pitch_hand"].agg(
        lambda h: h.iloc[0] if h.nunique() == 1 else "S").reset_index()
    arm = raw["arm_angle"].rename(columns={"pitcher": "pitcher_id", "year": "season", "ball_angle": "arm_angle"})
    arm = arm[["pitcher_id", "season", "arm_angle"]].drop_duplicates(key)
    base = base.merge(ah, on=key, how="left").merge(arm, on=key, how="left")
    piv = long.pivot(index=key, columns="pitch_type", values=list(WIDE_METRICS))
    cols = [(m, pt) for pt in PITCH_TYPES for m in WIDE_METRICS if (m, pt) in piv.columns]
    piv = piv[cols]
    piv.columns = [f"{pt.lower()}_{m}" for m, pt in cols]
    wide = base.merge(piv.reset_index(), on=key, how="left")
    return wide.sort_values(key).reset_index(drop=True)


CHANGE_METRICS = ("usage_pct", "avg_speed", "avg_spin", "break_x", "break_z_induced", "whiff_pct", "run_value_per_100")


def build_changes(long: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher, season and pitch type, for pitchers who also pitched the season before,
    covering every pitch type thrown in either of the two seasons."""
    cur = long[KEY + list(CHANGE_METRICS)]
    prev = cur.copy()
    prev["season"] += 1
    seasons = long[["pitcher_id", "season"]].drop_duplicates()
    both = seasons.merge(seasons.assign(season=seasons["season"] + 1), on=["pitcher_id", "season"])
    ch = cur.merge(prev, on=KEY, how="outer", suffixes=("", "_prev")).merge(both, on=["pitcher_id", "season"])
    names = long.sort_values("season").groupby("pitcher_id")["player_name"].last()
    ch["player_name"] = ch["pitcher_id"].map(names)
    ch["is_new"] = ch["usage_pct_prev"].isna()
    ch["is_dropped"] = ch["usage_pct"].isna()
    ch["usage_pct"] = ch["usage_pct"].where(~ch["is_dropped"], 0.0)
    ch["usage_pct_prev"] = ch["usage_pct_prev"].where(~ch["is_new"], 0.0)
    for m in CHANGE_METRICS:
        ch[f"{m}_delta"] = (ch[m] - ch[f"{m}_prev"]).round(3)
    ch["prev_season"] = ch["season"] - 1
    cols = ["pitcher_id", "player_name", "season", "prev_season", "pitch_type", "is_new", "is_dropped"]
    for m in CHANGE_METRICS:
        cols += [f"{m}_prev", m, f"{m}_delta"]
    return _sort(ch[cols])


# ----------------------------------------------------------------------------- gates

def _digest(df: pd.DataFrame, drop: set[str]) -> str:
    body = df.drop(columns=[c for c in df.columns if c in drop])
    body = body.reindex(sorted(body.columns), axis=1)
    rows = sorted(body.to_csv(index=False, header=False).splitlines())
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _near_same(a: pd.DataFrame, b: pd.DataFrame, key: list[str], drop: set[str]) -> str | None:
    """A season answered with another season's table (which Savant may since have revised a
    little) shares most keys with it and has identical numbers for most of them. Two real seasons
    share many pitchers but almost never the same numbers."""
    num = [c for c in a.columns if c not in drop and c not in key and pd.api.types.is_numeric_dtype(a[c])]
    a = a.drop_duplicates(key).set_index(key)[num].round(3)
    b = b.drop_duplicates(key).set_index(key)[num].round(3)
    shared = a.index.intersection(b.index)
    if not len(shared) or len(shared) < NEAR_SHARED * min(len(a), len(b)):
        return None
    same = ((a.loc[shared] == b.loc[shared]) | (a.loc[shared].isna() & b.loc[shared].isna())).all(axis=1)
    if same.mean() > NEAR_EQUAL:
        return f"{same.mean():.0%} of {len(shared)} shared {'/'.join(key)} have identical numbers"
    return None


def distinct_seasons(name: str, df: pd.DataFrame, season: str, key: list[str]) -> list[str]:
    errors = []
    drop = {season, "year", "season", "prev_season"}
    s = pd.to_numeric(df[season], errors="coerce")
    parts = {y: df[s == y] for y in YEARS}
    digests = {y: _digest(p, drop) for y, p in parts.items() if len(p)}
    for a, b in itertools.combinations(sorted(digests), 2):
        if digests[a] == digests[b]:
            errors.append(f"{name}: {a} and {b} are the same table")
        else:
            near = _near_same(parts[a], parts[b], key, drop)
            if near:
                errors.append(f"{name}: {a} and {b} are nearly the same table: {near}")
    return errors


def gates(raw: dict[str, pd.DataFrame], long: pd.DataFrame, wide: pd.DataFrame, changes: pd.DataFrame,
          old: pd.DataFrame | None) -> list[str]:
    errors: list[str] = []
    say = print

    # 1. seasons, exactly 2020-2026, in every table (raw and output)
    for name, df, col in [("stats", raw["stats"], "year"), ("arsenal", raw["arsenal"], "year"),
                          ("movement", raw["movement"], "year"), ("arm_angle", raw["arm_angle"], "year"),
                          ("pitcher_arsenal", long, "season"), ("pitcher_arsenal_wide", wide, "season")]:
        s = pd.to_numeric(df[col], errors="coerce")
        got = set(s.dropna().astype(int))
        if got != set(YEARS) or s.isna().any():
            errors.append(f"{name}: seasons {sorted(got)} != {list(YEARS)}")
    got = set(changes["season"])
    if got != set(YEARS[1:]):
        errors.append(f"arsenal_changes: seasons {sorted(got)} != {list(YEARS[1:])}")

    # 2. rows per season
    per = long.groupby("season").size()
    pitchers = wide.groupby("season").size()
    chg = changes.groupby("season").size()
    say(f"rows per season  long {per.to_dict()}\n                 wide {pitchers.to_dict()}\n"
        f"              changes {chg.to_dict()}")
    for y in YEARS:
        if per.get(y, 0) < MIN_ROWS_LONG:
            errors.append(f"pitcher_arsenal {y}: {per.get(y, 0)} rows < {MIN_ROWS_LONG}")
        if pitchers.get(y, 0) < MIN_PITCHERS:
            errors.append(f"pitcher_arsenal_wide {y}: {pitchers.get(y, 0)} pitchers < {MIN_PITCHERS}")
        if y > YEARS[0] and chg.get(y, 0) < MIN_CHANGES:
            errors.append(f"arsenal_changes {y}: {chg.get(y, 0)} rows < {MIN_CHANGES}")

    # 3. no two seasons the same or nearly the same
    errors += distinct_seasons("stats", raw["stats"], "year", ["player_id", "pitch_type"])
    errors += distinct_seasons("arsenal", raw["arsenal"], "year", ["pitcher_id", "pitch_type"])
    errors += distinct_seasons("movement", raw["movement"], "year", ["pitcher_id", "pitch_type"])
    errors += distinct_seasons("arm_angle", raw["arm_angle"], "year", ["pitcher"])
    errors += distinct_seasons("pitcher_arsenal", long, "season", ["pitcher_id", "pitch_type"])

    # 4. keys, names, pitch types
    if long.duplicated(KEY).any():
        errors.append("pitcher_arsenal: duplicate pitcher_id/season/pitch_type")
    if wide.duplicated(["pitcher_id", "season"]).any():
        errors.append("pitcher_arsenal_wide: duplicate pitcher_id/season")
    names = long["player_name"]
    bad_names = names.isna() | ~names.astype(str).str.fullmatch(r"[^,]+, .+")
    lower = names.astype(str).str.fullmatch(r"[^A-Z]*")
    if bad_names.any() or lower.any():
        errors.append(f"pitcher_arsenal: {int(bad_names.sum())} names not 'Last, First', {int(lower.sum())} all lowercase")
    extra = set(long["pitch_type"]) - set(PITCH_TYPES)
    if extra:
        errors.append(f"pitcher_arsenal: unexpected pitch types {sorted(extra)}")
    counts = long.pivot_table(index="season", columns="pitch_type", values="pitcher_id", aggfunc="size").fillna(0).astype(int)
    say("pitchers per pitch type and season:\n" + counts.reindex(columns=[p for p in PITCH_TYPES if p in counts]).to_string())
    for y in YEARS:
        have = set(long.loc[long["season"] == y, "pitch_type"])
        missing = CORE_PITCH_TYPES - have
        if missing:
            errors.append(f"{y}: core pitch types missing {sorted(missing)}")

    # 5. usage sums per pitcher-season (pitch types Savant does not list, e.g. eephus, are part of
    #    the denominator but have no row, so a sum can fall short of 100)
    u = wide.loc[wide["total_pitches"] >= USAGE_MIN_PITCHES, "usage_pct_sum"]
    bad = ((u < USAGE_LO) | (u > USAGE_HI)).mean()
    say(f"usage sum (pitchers with {USAGE_MIN_PITCHES}+ pitches, n={len(u)}): min {u.min():.1f} p1 {u.quantile(.01):.1f} "
        f"median {u.median():.1f} max {u.max():.1f}; outside [{USAGE_LO}, {USAGE_HI}]: {bad:.2%}")
    if len(u) == 0 or bad > USAGE_MAX_BAD:
        errors.append(f"usage sums: {bad:.2%} of pitcher-seasons outside [{USAGE_LO}, {USAGE_HI}] > {USAGE_MAX_BAD:.1%}")

    # 6. whiff% range and per-type medians
    w = long["whiff_pct"].dropna()
    if (w < 0).any() or (w > 100).any():
        errors.append("whiff_pct outside 0-100")
    q = long[long["pitches"] >= 100]
    med = q.pivot_table(index="season", columns="pitch_type", values="whiff_pct", aggfunc="median")
    say("median whiff_pct (pitch types thrown 100+ times):\n" + med.reindex(columns=list(WHIFF_MEDIAN)).round(1).to_string())
    for pt, (lo, hi) in WHIFF_MEDIAN.items():
        for y in YEARS:
            v = med[pt].get(y, np.nan) if pt in med else np.nan
            if not (lo <= v <= hi):
                errors.append(f"median whiff_pct {pt} {y} = {v} outside [{lo}, {hi}]")

    # 7. join coverage (rows of Pitch Arsenal Stats that got the other sources)
    for y in YEARS:
        part = long[(long["season"] == y) & long["in_arsenal_stats"]]
        ca = part["avg_speed"].notna().mean()
        cs = part["avg_spin"].notna().mean()
        cb = part["break_x"].notna().mean()
        cm = part["break_z_vs_similar"].notna().mean()
        cmq = part.loc[part["pitches"] >= 250, "break_z_vs_similar"].notna().mean()
        carm = wide.loc[wide["season"] == y, "arm_angle"].notna().mean()
        say(f"coverage {y}: speed {ca:.3f} spin {cs:.3f} break {cb:.3f} vs-similar {cm:.3f} "
            f"(pitch types thrown 250+ times {cmq:.3f}); arm angle (pitchers) {carm:.3f}")
        for label, v, floor in (("speed", ca, COVER_ARSENAL), ("spin", cs, COVER_ARSENAL), ("break", cb, COVER_ARSENAL),
                                ("vs-similar", cm, COVER_MOVEMENT), ("vs-similar 250+", cmq, COVER_MOVEMENT_250),
                                ("arm angle", carm, COVER_ARM)):
            if not v >= floor:
                errors.append(f"{y}: {label} coverage {v:.3f} < {floor}")

    # 8. the previous Kaggle version: its pitcher-seasons should be here
    if old is not None:
        o = old.dropna(subset=["player_id"])
        o = o[o["season"].isin(YEARS)][["player_id", "season"]].astype(int).drop_duplicates()
        k = set(map(tuple, wide[["pitcher_id", "season"]].to_numpy().tolist()))
        found = np.mean([tuple(r) in k for r in o.to_numpy().tolist()])
        say(f"old version pitcher-seasons found in the new wide table: {found:.3f} of {len(o)}")
        if not found >= OLD_COVER:
            errors.append(f"only {found:.3f} of the old version's pitcher-seasons are in the new build (< {OLD_COVER})")
    return errors


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carry", type=Path, required=True, help="the current Kaggle version, unzipped")
    ap.add_argument("--meta", type=Path, required=True, help="dataset-metadata.json with {rows:...} placeholders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, help="development only: reuse/store the fetched raw tables here")
    args = ap.parse_args()
    print(f"savant-extras {sx.__version__}")

    known = {p.name for p in args.carry.iterdir() if p.is_file()} - {"dataset-metadata.json"}
    if known - REPLACED:
        print(f"the current version has files this build does not replace: {sorted(known - REPLACED)}")
        return 1
    # The kaggle CLI can exit 0 without downloading anything: an empty carry must not pass.
    if OLD_FILE not in known and not {f"{n}.csv" for n in OUTPUTS} <= known:
        print(f"{args.carry} has neither {OLD_FILE} nor this build's CSVs: the current version was not downloaded")
        return 1
    # First rebuild: the old file's pitcher-seasons must be covered. Later rebuilds: the previous
    # build's wide table plays that role.
    old_path = args.carry / OLD_FILE
    if old_path.exists():
        old = pd.read_csv(old_path)
    else:
        old = pd.read_csv(args.carry / "pitcher_arsenal_wide.csv").rename(columns={"pitcher_id": "player_id"})

    raw = fetch_all(args.cache)
    long = build_long(raw)
    wide = build_wide(long, raw)
    changes = build_changes(long)
    errors = gates(raw, long, wide, changes, old)
    if errors:
        print("\n".join(["", "Gates failed, nothing written:"] + errors))
        return 1
    return write(args, {"pitcher_arsenal": long, "pitcher_arsenal_wide": wide, "arsenal_changes": changes})


def write(args, tables: dict[str, pd.DataFrame]) -> int:
    """{rows:<table>} is the table's total, {rows:<table>:<season>} one season."""
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    desc = meta["description"]
    for name, df in tables.items():
        per = df.groupby("season").size()
        desc = desc.replace("{rows:" + name + "}", f"{len(df):,}")
        for y in YEARS:
            desc = desc.replace("{rows:" + name + f":{y}" + "}", f"{int(per.get(y, 0)):,}")
    left = re.findall(r"\{rows:[a-z_:0-9]+\}", desc)
    if left:
        print(f"description placeholders with no table: {left}")
        return 1
    meta["description"] = desc
    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(args.out / f"{name}.csv", index=False)
    (args.out / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(tables)} CSVs and dataset-metadata.json to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
