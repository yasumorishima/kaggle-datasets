"""Build the CSVs of the Kaggle dataset "Baseball Savant Leaderboards (2024-2026)".

Run by `.github/workflows/update-dataset.yml` (via build.sh) before `kaggle datasets version`.
Nothing here is run locally.

Sources
- 16 Baseball Savant leaderboards through savant-extras 0.6.0, plus park_factors (Savant's
  Statcast park factors, new in 0.6.0). Through 0.5.0 several functions sent parameter names Savant
  ignores, and Savant then returns the current season.
- outs_above_average and outfield_jump straight from Savant's CSV endpoints. Savant leaves the
  `year` column of outs_above_average empty, so `year` is always written from the request.
- pitcher_quality (FanGraphs Stuff+, 2024-2025 only) is carried over unchanged from the current
  Kaggle version (`--carry`): FanGraphs refuses GitHub-hosted runners.

Every table goes through the same gates and the build fails (writing nothing) if any is violated:
the seasons present are exactly the expected ones, a minimum row count per season, no two season
slices are the same table (the failure mode of a request Savant did not understand is the current
season returned for every year asked), pitch_movement has the core pitch types in every season,
and every CSV of the current Kaggle version has a successor.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import savant_extras as sx

YEARS = (2024, 2025, 2026)
# Tables that do not cover every season, and why.
TABLE_YEARS = {
    # Savant lists no players for year=2026 (its qualified list for the latest season is empty);
    # the 2024 and 2025 lists still carry a column for every season 2015-2026.
    "year_to_year": (2024, 2025),
    # FanGraphs, carried over: not reachable from GitHub-hosted runners.
    "pitcher_quality": (2024, 2025),
}
# Pitch types every season must have in pitch_movement.
CORE_PITCH_TYPES = {"FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"}
# Every pitch type Savant's pitch movement leaderboard serves; each is one request.
PITCH_TYPES = ("FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO", "SC", "KN")
# Columns that name the season rather than measure anything (dropped before the duplicate check).
SEASON_COLS = {"year", "season", "start_year", "end_year"}
CARRIED = {"pitcher_quality": "pitcher_quality_2024_2025.csv"}
SLEEP = 1.0
UA = {"User-Agent": "kaggle-datasets-build/1.0"}


def season_dates(year: int) -> tuple[str, str]:
    """Regular-season start and end from MLB StatsAPI (so openers abroad are included)."""
    r = requests.get(f"https://statsapi.mlb.com/api/v1/seasons/{year}?sportId=1", timeout=30, headers=UA)
    r.raise_for_status()
    s = r.json()["seasons"][0]
    return s["regularSeasonStartDate"], s["regularSeasonEndDate"]


def _paced(fn, *args, **kwargs) -> pd.DataFrame:
    time.sleep(SLEEP)
    return fn(*args, **kwargs)


def _per_year(fn, years, **kwargs) -> pd.DataFrame:
    frames = []
    for y in years:
        df = _paced(fn, y, **kwargs)
        # A `season` column (park_factors) must be the season asked for; `year` replaces it.
        if "season" in df.columns:
            got = set(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int))
            if got != {y}:
                raise SystemExit(f"{fn.__name__}: asked for {y}, got seasons {sorted(got)}")
            df = df.drop(columns="season")
        # Always overwrite: Savant sometimes returns a `year` column that is empty.
        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _bat_tracking(years) -> pd.DataFrame:
    frames = []
    for y in years:
        start, end = season_dates(y)
        df = _paced(sx.bat_tracking, start, end, min_swings=100)
        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _pitch_movement(years) -> pd.DataFrame:
    frames = []
    for y in years:
        for pt in PITCH_TYPES:
            # An empty answer is a warning (EmptySavantResponse), not an exception, and gives an
            # empty frame. Rare types may be absent; the core types are required by check().
            df = _paced(sx.pitch_movement, y, pitch_type=pt)
            if df.empty:
                continue
            df["pitch_type"] = pt
            df["year"] = y
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _savant_csv(url: str) -> pd.DataFrame:
    time.sleep(SLEEP)
    r = requests.get(url, timeout=60, headers=UA)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))


def _outs_above_average(years) -> pd.DataFrame:
    frames = []
    for y in years:
        df = _savant_csv(
            "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
            f"?type=Fielder&startYear={y}&endYear={y}&split=no&team=&range=year"
            "&min=q&pos=&roles=&viz=hide&csv=true"
        )
        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _outfield_jump(years) -> pd.DataFrame:
    frames = []
    for y in years:
        df = _savant_csv(f"https://baseballsavant.mlb.com/leaderboard/outfield_jump?year={y}&min=q&csv=true")
        # Savant fills `year` here; a different one means the parameter was not read.
        got = set(pd.to_numeric(df.get("year"), errors="coerce").dropna().astype(int)) if "year" in df else set()
        if got and got != {y}:
            raise SystemExit(f"outfield_jump: asked for {y}, Savant returned seasons {sorted(got)}")
        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _pitch_tempo(df: pd.DataFrame) -> pd.DataFrame:
    """Savant's pitch-tempo CSV repeats two header names: total_pitches, and median_seconds_empty in
    the place of the runners-on median. In every row checked (2024-2026) the repeated columns hold
    the same values as the first ones, i.e. the export carries no runners-on median. Drop the
    copies; if Savant ever fills the second one with different values, keep it under its real name."""
    df = df.drop(columns="total_pitches.1") if (df["total_pitches.1"] == df["total_pitches"]).all() else df
    if (df["median_seconds_empty.1"] == df["median_seconds_empty"]).all():
        return df.drop(columns="median_seconds_empty.1")
    return df.rename(columns={"median_seconds_empty.1": "median_seconds_onbase"})


def _y(fn, **kwargs):
    return lambda years: _per_year(fn, years, **kwargs)


BUILDERS = {
    "arm_strength": _y(sx.arm_strength),
    "baserunning": _y(sx.baserunning),
    "basestealing": _y(sx.basestealing),
    "bat_tracking": _bat_tracking,
    "batted_ball": _y(sx.batted_ball),
    "catcher_blocking": _y(sx.catcher_blocking),
    "catcher_stance": _y(sx.catcher_stance),
    "catcher_throwing": _y(sx.catcher_throwing),
    "home_runs": _y(sx.home_runs),
    "pitch_movement": _pitch_movement,
    "pitch_tempo": lambda years: _pitch_tempo(_per_year(sx.pitch_tempo, years)),
    "pitcher_arm_angle": _y(sx.pitcher_arm_angle),
    "running_game": _y(sx.running_game),
    "swing_take": _y(sx.swing_take),
    "timer_infractions": _y(sx.timer_infractions),
    # `year` here is the season whose qualified players are listed; the columns span 2015-now.
    "year_to_year": _y(sx.year_to_year),
    "park_factors": _y(sx.park_factors),
    "outs_above_average": _outs_above_average,
    "outfield_jump": _outfield_jump,
}

# Minimum rows per season. Team-level tables have 30; the rest are player tables.
MIN_ROWS = {"park_factors": 30, "timer_infractions": 30}
DEFAULT_MIN_ROWS = 40


def years_of(name: str) -> tuple[int, ...]:
    return TABLE_YEARS.get(name, YEARS)


def _season_col(name: str, df: pd.DataFrame) -> str:
    for c in ("year", "season"):
        if c in df.columns:
            return c
    raise SystemExit(f"{name}: no year/season column")


def _slice_digest(df: pd.DataFrame) -> str:
    body = df.drop(columns=[c for c in df.columns if c in SEASON_COLS])
    body = body.reindex(sorted(body.columns), axis=1)
    rows = sorted(body.to_csv(index=False, header=False).splitlines())
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


# Keys for the near-duplicate check, first match wins.
KEYS = (("pitcher_id", "pitch_type"), ("player_id",), ("resp_fielder_id",), ("pitcher",), ("entity_id",), ("id",),
        ("name", "team"), ("team",), ("name",))
# Tables where the near check does not apply: year_to_year lists every season as its own column, so
# a batter qualified in both 2024 and 2025 has the same numbers in both lists by construction (an
# identical table is still caught by the exact digest).
NO_NEAR_CHECK = {"year_to_year"}
NEAR_SHARED = 0.9   # share of the smaller season's keys also present in the other season
NEAR_EQUAL = 0.9    # share of those shared keys whose numeric columns all agree


def _near_same(a: pd.DataFrame, b: pd.DataFrame) -> str | None:
    """Savant revises past seasons a little, so a season answered with another season's table can
    differ from it in a few cells. Two seasons of a real leaderboard share many players but almost
    never the same numbers for most of them; flag that."""
    key = next((list(k) for k in KEYS if all(c in a.columns for c in k)), None)
    if key is None:
        return None
    if a.duplicated(key).any() or b.duplicated(key).any():
        return f"key {'/'.join(key)} is not unique, the near check cannot compare rows"
    num = [c for c in a.columns if c not in SEASON_COLS and c not in key and pd.api.types.is_numeric_dtype(a[c])]
    if not num:
        return None
    a = a.drop_duplicates(key).set_index(key)[num].round(3)
    b = b.drop_duplicates(key).set_index(key)[num].round(3)
    shared = a.index.intersection(b.index)
    if len(shared) < NEAR_SHARED * min(len(a), len(b)):
        return None
    same = ((a.loc[shared] == b.loc[shared]) | (a.loc[shared].isna() & b.loc[shared].isna())).all(axis=1)
    if same.mean() > NEAR_EQUAL:
        return f"{same.mean():.0%} of {len(shared)} shared {'/'.join(key)} have identical numbers"
    return None


def check(name: str, df: pd.DataFrame) -> list[str]:
    errors = []
    want = years_of(name)
    col = _season_col(name, df)
    seasons = pd.to_numeric(df[col], errors="coerce")
    if seasons.isna().any():
        errors.append(f"{int(seasons.isna().sum())} rows with no {col}")
    got = set(seasons.dropna().astype(int))
    if got != set(want):
        errors.append(f"seasons {sorted(got)} != {list(want)}")
    floor = MIN_ROWS.get(name, DEFAULT_MIN_ROWS)
    parts = {y: df[seasons == y] for y in want}
    for y, part in parts.items():
        if len(part) < floor:
            errors.append(f"{y}: {len(part)} rows < {floor}")
    digests = {y: _slice_digest(p) for y, p in parts.items() if len(p)}
    for a, b in itertools.combinations(sorted(digests), 2):
        if digests[a] == digests[b]:
            errors.append(f"{a} and {b} are the same table")
        elif name not in NO_NEAR_CHECK:
            near = _near_same(parts[a], parts[b])
            if near:
                errors.append(f"{a} and {b} are nearly the same table: {near}")
    if name == "pitch_movement":
        for y in want:
            missing = CORE_PITCH_TYPES - set(df.loc[seasons == y, "pitch_type"])
            if missing:
                errors.append(f"{y}: missing pitch types {sorted(missing)}")
    if name == "year_to_year":
        # The list for season Y is Y's qualified batters, so each has an xwOBA in column Y
        # (100% in 2024 and 2025). Another season's list relabeled as Y has batters who did not
        # play in Y (measured: 2025's list has a 2024 value for 94%, 2024's a 2025 value for 98%).
        for y in want:
            if str(y) in df.columns:
                share = df.loc[seasons == y, str(y)].notna().mean()
                if share < 1:
                    errors.append(f"{y}: only {share:.1%} of the listed batters have a {y} xwOBA")
        missing = [str(y) for y in YEARS if str(y) not in df.columns]
        if missing:
            errors.append(f"no column for seasons {missing}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carry", type=Path, required=True, help="the current Kaggle version, unzipped")
    ap.add_argument("--meta", type=Path, required=True, help="dataset-metadata.json with {rows:<table>} placeholders")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"savant-extras {sx.__version__}")
    tables: dict[str, pd.DataFrame] = {}
    for name, build in BUILDERS.items():
        tables[name] = build(years_of(name))
    for name, fname in CARRIED.items():
        path = args.carry / fname
        if not path.exists():
            raise SystemExit(f"{name}: {path} not found in the carried-over files")
        tables[name] = pd.read_csv(path)

    new = {f"{n}.csv" for n in tables}
    failures = []
    # Every CSV of the current version must have a successor (`<table>_2024_2025.csv` -> `<table>.csv`).
    # The column-description helper scripts (.js) and file_descriptions.txt are no longer shipped.
    old_csv = sorted(p.name for p in args.carry.iterdir() if p.suffix == ".csv")
    for f in old_csv:
        succ = re.sub(r"_\d{4}_\d{4}\.csv$", ".csv", f)
        if succ not in new:
            failures.append(f"{f} in the current version has no successor {succ}")
    for name in sorted(tables):
        df = tables[name]
        errs = check(name, df)
        col = _season_col(name, df)
        per = df[col].value_counts().sort_index().to_dict()
        print(f"{'FAIL' if errs else 'ok  '} {name:<20} {df.shape} {per}")
        failures += [f"{name}: {e}" for e in errs]
    if failures:
        print("\n".join(["", "Gates failed, nothing written:"] + failures))
        return 1

    # The description's row-count column is filled from what was actually built.
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    desc = meta["description"]
    for name, df in tables.items():
        per = df[_season_col(name, df)].value_counts()
        cell = " / ".join(f"{int(per[y]):,}" if y in per else "-" for y in YEARS)
        desc = desc.replace("{rows:" + name + "}", cell)
    desc = desc.replace("{build_date}", time.strftime("%Y-%m-%d", time.gmtime()))
    left = re.findall(r"\{rows:[a-z_]+\}|\{build_date\}", desc)
    if left:
        print(f"description placeholders with no table: {left}")
        return 1
    missing = [n for n in tables if f"`{n}.csv`" not in desc]
    if missing:
        print(f"tables not in the description: {missing}")
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
