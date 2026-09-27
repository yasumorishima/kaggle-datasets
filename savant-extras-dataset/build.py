"""Build the 20 CSVs of the Kaggle dataset "Baseball Savant Leaderboards (2024-2025)".

Run by `.github/workflows/update-dataset.yml` (via build.sh) before `kaggle datasets version`.
Nothing here is run locally.

Sources
- 16 Baseball Savant leaderboards through savant-extras 0.6.0. Through 0.5.0 several functions
  sent parameter names Savant ignores, and Savant then returns the current season: the previous
  version of this dataset had 7 tables where 2024 and 2025 were the same table, an empty
  swing_take, a 2-row league-level catcher_stance and a four-seam-only pitch_movement.
- outs_above_average straight from Savant's CSV endpoint. Savant leaves its `year` column empty,
  and the old notebook only filled `year` when the column was missing, so every row had year NaN.
- outfield_jump, pitcher_quality (FanGraphs Stuff+) and park_factors (FanGraphs) are carried over
  unchanged from the current Kaggle version (`--carry`): FanGraphs refuses GitHub-hosted runners,
  and these three were checked to differ between 2024 and 2025. They still pass the gates below.

Every table goes through the same gates and the build fails (writing nothing) if any is violated:
seasons present are exactly the expected ones, a minimum row count per season, the 2024 and 2025
slices must not be the same table, neither may equal the same table fetched for the current
season, pitch_movement must have the core pitch types in both seasons, and no file of the current
Kaggle version may be dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import savant_extras as sx

YEARS = (2024, 2025)
# Regular seasons (Seoul opener 2024-03-20, Tokyo opener 2025-03-18).
SEASON_DATES = {2024: ("2024-03-20", "2024-09-30"), 2025: ("2025-03-18", "2025-09-28")}
# The failure this build guards against is Savant answering with the current season. Every fetched
# table is also fetched for CURRENT, and a 2024 or 2025 slice equal to it fails the build. (Six
# tables are not season-checked by savant-extras: arm_strength, pitch_tempo and pitcher_arm_angle
# have no season column, year_to_year never checks, bat_tracking is by date, OAA is a direct URL.)
CURRENT = 2026
SEASON_DATES[CURRENT] = ("2026-03-01", "2026-11-30")
# Pitch types every season must have in pitch_movement.
CORE_PITCH_TYPES = {"FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"}
# Every pitch type Savant's pitch movement leaderboard serves; each is one request.
PITCH_TYPES = ("FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO", "SC", "KN")
# Columns that name the season rather than measure anything (dropped before the duplicate check).
SEASON_COLS = {"year", "season", "start_year", "end_year"}
CARRIED = ("outfield_jump", "pitcher_quality", "park_factors")
SLEEP = 1.0


def _paced(fn, *args, **kwargs) -> pd.DataFrame:
    time.sleep(SLEEP)
    return fn(*args, **kwargs)


def _per_year(fn, years, **kwargs) -> pd.DataFrame:
    frames = []
    for y in years:
        df = _paced(fn, y, **kwargs)
        # Always overwrite: Savant sometimes returns a `year` column that is empty.
        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _bat_tracking(years) -> pd.DataFrame:
    frames = []
    for y in years:
        start, end = SEASON_DATES[y]
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
            if "pitch_type" in df.columns:
                got = set(df["pitch_type"].dropna().astype(str))
                if got and got != {pt}:
                    raise SystemExit(f"pitch_movement {y} {pt}: Savant returned pitch types {sorted(got)}")
            df["pitch_type"] = pt
            df["year"] = y
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _outs_above_average(years) -> pd.DataFrame:
    frames = []
    for y in years:
        time.sleep(SLEEP)
        url = (
            "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
            f"?type=Fielder&startYear={y}&endYear={y}&split=no&team=&range=year"
            "&min=q&pos=&roles=&viz=hide&csv=true"
        )
        r = requests.get(url, timeout=60, headers={"User-Agent": "kaggle-datasets-build/1.0"})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _carried(name: str, carry: Path) -> pd.DataFrame:
    path = carry / f"{name}_2024_2025.csv"
    if not path.exists():
        raise SystemExit(f"{name}: {path} not found in the carried-over files")
    return pd.read_csv(path)


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
    "pitch_tempo": _y(sx.pitch_tempo),
    "pitcher_arm_angle": _y(sx.pitcher_arm_angle),
    "running_game": _y(sx.running_game),
    "swing_take": _y(sx.swing_take),
    "timer_infractions": _y(sx.timer_infractions),
    # `year` here is the season whose qualified players are listed; the columns span 2015-now.
    "year_to_year": _y(sx.year_to_year),
    "outs_above_average": _outs_above_average,
}

# Minimum rows per season. Team-level tables have 30; the rest are player tables.
MIN_ROWS = {"park_factors": 30, "timer_infractions": 30}
DEFAULT_MIN_ROWS = 40


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


def check(name: str, df: pd.DataFrame, current: pd.DataFrame | None = None) -> list[str]:
    errors = []
    col = _season_col(name, df)
    seasons = pd.to_numeric(df[col], errors="coerce")
    if seasons.isna().any():
        errors.append(f"{int(seasons.isna().sum())} rows with no {col}")
    got = set(seasons.dropna().astype(int))
    if got != set(YEARS):
        errors.append(f"seasons {sorted(got)} != {list(YEARS)}")
    floor = MIN_ROWS.get(name, DEFAULT_MIN_ROWS)
    for y in YEARS:
        n = int((seasons == y).sum())
        if n < floor:
            errors.append(f"{y}: {n} rows < {floor}")
    a, b = (df[seasons == y] for y in YEARS)
    if len(a) and len(b) and _slice_digest(a) == _slice_digest(b):
        errors.append("2024 and 2025 are the same table")
    if current is not None and len(current):
        cur = _slice_digest(current)
        for y, part in zip(YEARS, (a, b)):
            if len(part) and _slice_digest(part) == cur:
                errors.append(f"{y} is the same table as the current season {CURRENT}")
    if name == "pitch_movement":
        for y in YEARS:
            missing = CORE_PITCH_TYPES - set(df.loc[seasons == y, "pitch_type"])
            if missing:
                errors.append(f"{y}: missing pitch types {sorted(missing)}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carry", type=Path, required=True, help="the current Kaggle version, unzipped")
    ap.add_argument("--meta", type=Path, required=True, help="dataset-metadata.json with {rows:<table>} placeholders")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"savant-extras {sx.__version__}")
    tables: dict[str, pd.DataFrame] = {}
    currents: dict[str, pd.DataFrame] = {}
    for name, build in BUILDERS.items():
        tables[name] = build(YEARS)
        currents[name] = build((CURRENT,))
    for name in CARRIED:
        tables[name] = _carried(name, args.carry)

    # Non-CSV files of the dataset (column-description scripts, file_descriptions.txt) ship too,
    # from the repo so that edits to them are published.
    extras = sorted(p for p in args.meta.parent.iterdir() if p.suffix in (".js", ".txt"))
    old = {p.name for p in args.carry.iterdir() if p.is_file()}
    new = {f"{n}_2024_2025.csv" for n in tables} | {p.name for p in extras}
    failures = []
    if old - new:
        failures.append(f"files in the current version that the build would drop: {sorted(old - new)}")
    for name in sorted(tables):
        df = tables[name]
        errs = check(name, df, currents.get(name))
        col = _season_col(name, df) if ("year" in df or "season" in df) else None
        per = df[col].value_counts().sort_index().to_dict() if col else {}
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
        desc = desc.replace("{rows:" + name + "}", " / ".join(f"{int(per.get(y, 0)):,}" for y in YEARS))
    left = re.findall(r"\{rows:[a-z_]+\}", desc)
    if left:
        print(f"description placeholders with no table: {left}")
        return 1
    meta["description"] = desc

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(args.out / f"{name}_2024_2025.csv", index=False)
    for p in extras:
        shutil.copy2(p, args.out / p.name)
    (args.out / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(tables)} CSVs, {len(extras)} description files and dataset-metadata.json to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
