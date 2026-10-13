"""Build the private Kaggle dataset "MLB Statcast 2021-2023 pitch subset"
(yasunorim/mlb-statcast-2021-2023-pitch-subset): the input of a pre-registered confirmatory test of the
pitcher spin-axis gap study (kaggle-datasets commit e83a56e), which runs the frozen code on MLB
2021-2023 regular-season pitches.

Run by `.github/workflows/update-dataset.yml` (via build.sh). Writes _build/statcast_{2021,2022,2023}.parquet
with exactly the 23 columns the frozen pitcher notebooks read (COLUMNS), typed as in the frozen
dataset4 files (yasunorim/mlb-statcast-bat-tracking-2024-2025), plus tracking_gaps.csv and
dataset-metadata.json. Nothing about injured-list placements or transactions is read here.

The fetch and parse functions are imported unchanged from dataset4_statcast_bat_tracking/build.py
(one Savant statcast_search CSV request per calendar day, regular-season games only, the same
User-Agent, SLEEP and retries; StatsAPI /seasons, /teams, /schedule). Its per-game completeness check
(incomplete_games, partial_columns) is reused with this dataset's configuration set on the imported
module (see CONFIG below): the per-game "column empty in the whole game" rule looks only at the kept
columns, the per-game fill floors only at the spin and extension columns, and the 2024-2026 exception
tables are empty.

Gates (the build fails and writes nothing if any is violated):
- every day's response (dataset4's parse_day): Savant's known header, game_date is the requested day,
  game_type R only, game_year is the season, fewer than DAY_MAX rows, known pitch descriptions,
  integer columns hold integers, zone in 1-9/11-14;
- on days with at least NULL_CHECK_MIN_ROWS pitches at home venues, no kept column (arm_angle
  excepted) is empty for the whole day, and in every game with at least GAME_CHECK_MIN_PITCHES
  pitches no kept column (arm_angle excepted) is empty for the whole game; a pitch-tracking column
  (TRACKING) empty for a day or a game, or below its fill floor in a game, is a permanent tracking gap
  (every day here is years old, so nothing is still being backfilled): it is listed in
  tracking_gaps.csv, and the season fails if more than MAX_GAP_GAME_SHARE of its scheduled games
  have one;
- seasons are exactly 2021-2023, each with rows; per season: rows >= ROW_FLOOR and the rows that are
  not pitch-clock automatic balls/strikes equal Savant's own season total (group_by=team) [full
  build]; in a sample build the same comparison per fetched day; every regular-season game StatsAPI
  lists as Final on a fetched day has pitches, and no game outside the season's regular-season
  schedule does; no duplicate (game_pk, at_bat_number, pitch_number) within or across seasons; no two
  seasons with the same content; pitch_type filled on >= PITCH_TYPE_FLOOR of rows; the last
  regular-season date with a Final game (StatsAPI, and the data's last game_date in a full build) is
  in [Sep 15, Oct 6], and the first and last such dates have rows [when fetched]; no kept column but
  arm_angle is empty for the whole season [full build];
- the Arrow types are FROZEN_TYPES (written out; dataset4's SCHEMA must still agree, checked at import),
  and each written file's schema is re-read and compared with them.
arm_angle is printed (fill on thrown pitches) but not gated (filled on 99.4-99.6% of rows on the three
sample days 2021-06-01, 2022-07-15, 2023-09-20; a season-wide floor was not measured).
The build log ends with an md5 manifest of the three Parquet files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# The dataset4 builder, loaded under its own module name (this file is also called build.py, so a plain
# `import build` would import this file).
_D4 = Path(__file__).resolve().parent.parent / "dataset4_statcast_bat_tracking" / "build.py"
_spec = importlib.util.spec_from_file_location("dataset4_build", _D4)
d4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d4)

YEARS = (2021, 2022, 2023)
# The columns the frozen pitcher notebooks read (PCOLS of pitcher-anomaly-il at 2c6b01d plus spin_axis).
COLUMNS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number", "pitch_type", "release_speed",
           "release_spin_rate", "release_extension", "release_pos_x", "release_pos_z", "arm_angle", "pfx_x", "pfx_z",
           "zone", "description", "events", "type", "launch_speed", "woba_denom", "inning", "game_type", "spin_axis"]
assert len(COLUMNS) == 23 and len(set(COLUMNS)) == 23 and set(COLUMNS) <= set(d4.SAVANT_COLUMNS)
# The frozen files' Arrow types for COLUMNS, written out so that a later change of dataset4's SCHEMA fails
# here instead of changing these files (read from a dataset4-built statcast_2024.parquet, 2026-10-14).
FROZEN_TYPES = {
    "pitcher": pa.int64(), "game_date": pa.date32(), "game_pk": pa.int64(), "at_bat_number": pa.int64(),
    "pitch_number": pa.int64(), "pitch_type": pa.string(), "release_speed": pa.float64(),
    "release_spin_rate": pa.float64(), "release_extension": pa.float64(), "release_pos_x": pa.float64(),
    "release_pos_z": pa.float64(), "arm_angle": pa.float64(), "pfx_x": pa.float64(), "pfx_z": pa.float64(),
    "zone": pa.int64(), "description": pa.string(), "events": pa.string(), "type": pa.string(),
    "launch_speed": pa.float64(), "woba_denom": pa.int64(), "inning": pa.int64(), "game_type": pa.string(),
    "spin_axis": pa.float64(),
}
SCHEMA = pa.schema([pa.field(c, FROZEN_TYPES[c]) for c in COLUMNS])
assert SCHEMA.equals(pa.schema([d4.SCHEMA.field(c) for c in COLUMNS])), "dataset4's SCHEMA no longer matches the frozen types"
NOT_GATED = {"arm_angle"}
# Columns measured by pitch / batted-ball tracking: a whole game or day without them is a tracking gap.
TRACKING = {"pitch_type", "release_speed", "release_spin_rate", "release_extension", "release_pos_x",
            "release_pos_z", "pfx_x", "pfx_z", "spin_axis", "zone", "launch_speed"}
GAME_CHECKED = [c for c in COLUMNS if c not in NOT_GATED]
assert TRACKING <= set(GAME_CHECKED)
# Savant counted 709,852 (2021), 708,540 (2022) and 717,945 (2023) regular-season pitches with
# group_by=team (measured 2026-10-14).
ROW_FLOOR = 700_000
PITCH_TYPE_FLOOR = 0.90  # the frozen notebook asserts pitch_type >= 90% filled per season
LAST_DAY_WINDOW = ((9, 15), (10, 6))  # the frozen notebook asserts the season's last date in this window
MAX_GAP_GAME_SHARE = d4.MAX_GAP_GAME_SHARE
GAP_SETTLED_DAYS = d4.GAP_SETTLED_DAYS
GAP_COLUMNS = ["season", "game_date", "game_pk", "pitches", "empty_columns", "partial_columns"]

# CONFIG: dataset4's per-game check, configured for these seasons on the imported module (its functions
# read these module globals at call time; the module object is private to this script).
# - fill floors only for the kept spin / extension columns (dataset4's measured floors); bat tracking,
#   arm angle, effective speed and fielding alignment are not in this dataset;
d4.COLUMN_FLOORS = {c: d4.COLUMN_FLOORS[c] for c in ("release_spin_rate", "spin_axis", "release_extension")}
# - the 2024-2026 exception tables name games of other seasons;
d4.GAME_FLOOR_EXCEPTIONS = {}
d4.XWOBA_EXCEPTIONS = {}
d4.NEUTRAL_GAPS = {}
# - xwOBA and delta_run_exp are not in this dataset, so their per-game share floors are off.
d4.GAME_FLOORS = {"xwoba_share": 0.0, "dre_share": 0.0}


def savant_day_total(day: date) -> int:
    """Savant's own count of regular-season pitches on one day (group_by=team, as d4.savant_total)."""
    time.sleep(d4.SLEEP)
    r = d4._get(d4.SAVANT, {"all": "true", "hfGT": "R|", "hfSea": f"{day.year}|", "player_type": "pitcher",
                            "game_date_gt": day.isoformat(), "game_date_lt": day.isoformat(),
                            "group_by": "team", "min_pitches": 0, "min_results": 0, "sort_col": "pitches",
                            "sort_order": "desc"})
    text = r.content.decode("utf-8-sig")
    if not text.strip():  # no game that day
        return 0
    df = pd.read_csv(io.StringIO(text))
    if "pitches" not in df or len(df) > 30:
        raise SystemExit(f"Savant day total {day}: unexpected answer {df.shape}")
    return int(df["pitches"].sum())


def _row_hash(df: pd.DataFrame) -> np.ndarray:
    """Content without the season-naming columns (game_date, game_pk): the same pitches under another
    season would match."""
    body = df[["pitcher", "description", "release_speed", "pfx_x", "pfx_z", "launch_speed"]]
    return pd.util.hash_pandas_object(body, index=False).to_numpy()


def build_season(year: int, stage: Path, only_days: set[date] | None, cache: Path | None, full: bool):
    errors: list[str] = []
    start, end = d4.season_dates(year)
    sched = d4.schedule(year, start, end)
    final = sched[sched["state"] == "F"]
    final_days = sorted({date.fromisoformat(d) for d in final["official_date"]})
    first_day, last_day = final_days[0], final_days[-1]
    if date.today() - last_day <= timedelta(GAP_SETTLED_DAYS):
        raise SystemExit(f"{year}: season ended {last_day}; this builder assumes settled seasons")
    lo, hi = date(year, *LAST_DAY_WINDOW[0]), date(year, *LAST_DAY_WINDOW[1])
    if not lo <= last_day <= hi:
        errors.append(f"last regular-season day with a Final game (StatsAPI) {last_day} not in [{lo}, {hi}]")
    neutral = set(sched.loc[sched["neutral"], "game_pk"].astype(int))
    span0, span1 = min(start, first_day), max(end, last_day)
    days = [span0 + timedelta(n) for n in range((span1 - span0).days + 1)]
    if only_days is not None:
        days = [d for d in days if d in only_days]
    print(f"== {year}: regular season {start}..{end} (Final games {first_day}..{last_day}), "
          f"{sched['game_pk'].nunique()} games scheduled, {final['game_pk'].nunique()} Final, "
          f"{len(neutral)} at neutral sites, {len(days)} days to fetch", flush=True)
    writer = pq.ParquetWriter(stage / f"statcast_{year}.parquet", SCHEMA, compression="zstd")
    buf: list[pd.DataFrame] = []
    buffered = rows = thrown_rows = auto_rows = arm_thrown = 0
    keys, hashes, pks = [], [], set()
    nonnull = pd.Series(0, index=COLUMNS, dtype="int64")
    per_day: dict[date, int] = {}
    gaps: list[dict] = []
    t0 = time.time()
    try:
        for d in days:
            t = time.time()
            try:
                df = d4.parse_day(d4.fetch_day(d, cache), d, year)
                day_key = d4._key(df)
            except d4.GateError as e:  # recorded; the run goes on so one run lists every problem
                errors.append(str(e))
                per_day[d] = -1
                print(f"  {d} FAILED: {e}", flush=True)
                continue
            per_day[d] = len(df)
            n_auto = int(df["description"].isin(d4.AUTOMATIC).sum()) if len(df) else 0
            print(f"  {d} {len(df):>5} rows, {df['game_pk'].nunique() if len(df) else 0:>2} games, "
                  f"{time.time() - t:5.1f}s", flush=True)
            if not full:  # per-day form of the season-total gate
                total = savant_day_total(d)
                print(f"    Savant day total {total:,}; rows that are not automatic balls/strikes "
                      f"{len(df) - n_auto:,}", flush=True)
                if len(df) - n_auto != total:
                    errors.append(f"{d}: {len(df) - n_auto:,} rows that are not automatic balls/strikes but "
                                  f"Savant counts {total:,} pitches")
            if not len(df):
                continue
            nonnull += df[COLUMNS].notna().sum()
            thrown = df["pitch_type"].notna()
            thrown_rows += int(thrown.sum())
            arm_thrown += int((thrown & df["arm_angle"].notna()).sum())
            auto_rows += n_auto
            # Whole day, home venues only (neutral sites can lack tracking).
            home = df[~df["game_pk"].astype(int).isin(neutral)]
            day_gap_cols: list[str] = []
            if len(home) >= d4.NULL_CHECK_MIN_ROWS:
                empty = [c for c in GAME_CHECKED if home[c].isna().all()]
                hard = [c for c in empty if c not in TRACKING]
                day_gap_cols = [c for c in empty if c in TRACKING]
                if hard:
                    errors.append(f"{d}: {len(home)} pitches at home venues but empty {hard}")
            # Per game (dataset4's check; allowed_empty = every Savant column that is not kept-and-gated).
            low = d4.incomplete_games(df, set(d4.SAVANT_COLUMNS) - set(GAME_CHECKED))
            low = low[low["has_problem"]]
            listed = set()
            for r in low.to_dict("records"):
                parts = [x for x in r["partial_columns"] if not x.startswith("stale ")]
                cols = list(r["empty_columns"]) + [x.split()[0] for x in parts]
                if r["share_low"] or any(c not in TRACKING for c in cols):
                    errors.append(f"{d}: game {r['game_pk']} ({r['pitches']} pitches): empty {r['empty_columns']}, "
                                  f"partial {parts}")
                    continue
                listed.add(int(r["game_pk"]))
                gaps.append({"season": year, "game_date": d.isoformat(), "game_pk": int(r["game_pk"]),
                             "pitches": int(r["pitches"]), "empty_columns": ";".join(r["empty_columns"]),
                             "partial_columns": ";".join(parts)})
            if day_gap_cols:
                for pk, gm in home.groupby(home["game_pk"].astype("int64")):
                    if int(pk) not in listed:
                        gaps.append({"season": year, "game_date": d.isoformat(), "game_pk": int(pk),
                                     "pitches": len(gm), "empty_columns": ";".join(day_gap_cols),
                                     "partial_columns": ""})
            keys.append(day_key)
            hashes.append(_row_hash(df))
            pks.update(df["game_pk"].astype(int).unique().tolist())
            rows += len(df)
            buf.append(df[COLUMNS])
            buffered += len(df)
            if buffered >= 150_000:
                writer.write_table(pa.Table.from_pandas(pd.concat(buf, ignore_index=True), schema=SCHEMA,
                                                        preserve_index=False))
                buf, buffered = [], 0
        if buf:
            writer.write_table(pa.Table.from_pandas(pd.concat(buf, ignore_index=True), schema=SCHEMA,
                                                    preserve_index=False))
    finally:
        writer.close()
    print(f"  {year}: {rows:,} rows in {time.time() - t0:.0f}s", flush=True)

    key = np.concatenate(keys) if keys else np.array([], dtype="int64")
    fetched = set(days)
    if full:
        if rows < ROW_FLOOR:
            errors.append(f"{rows:,} rows < {ROW_FLOOR:,}")
        total = d4.savant_total(year)
        if rows - auto_rows != total:
            errors.append(f"{rows - auto_rows:,} rows that are not automatic balls/strikes but Savant counts "
                          f"{total:,} regular-season pitches")
    for d in (first_day, last_day):
        if d in fetched and per_day.get(d, 0) <= 0:
            errors.append(f"regular-season day {d} (first/last with a Final game) has no pitches")
    fin = final[final["official_date"].isin({x.isoformat() for x in fetched})]
    missing = sorted(set(fin["game_pk"].astype(int)) - pks)
    if missing:
        errors.append(f"{len(missing)} Final regular-season games with no pitches, e.g. {missing[:5]}")
    outside = sorted(pks - set(sched["game_pk"].astype(int)))
    if outside:
        errors.append(f"games not in the regular-season schedule: {outside[:5]}")
    if len(np.unique(key)) != len(key):
        errors.append(f"{len(key) - len(np.unique(key))} duplicate (game_pk, at_bat_number, pitch_number)")
    pt_fill = nonnull["pitch_type"] / rows if rows else 0.0
    if rows and pt_fill < PITCH_TYPE_FLOOR:
        errors.append(f"pitch_type filled on {pt_fill:.4f} < {PITCH_TYPE_FLOOR}")
    data_days = [d for d, n in per_day.items() if n > 0]
    if full and data_days and not (lo <= max(data_days) <= hi and max(data_days) == last_day):
        errors.append(f"last date with pitches {max(data_days)} is not {last_day} in [{lo}, {hi}]")
    empty_all = [c for c in GAME_CHECKED if nonnull[c] == 0]
    if full and empty_all:
        errors.append(f"columns empty in every row of the season: {empty_all}")
    n_sched = sched["game_pk"].nunique()
    n_gap = len({g["game_pk"] for g in gaps})
    if full and n_gap > MAX_GAP_GAME_SHARE * n_sched:
        errors.append(f"{n_gap} of {n_sched} games have tracking gaps (> {MAX_GAP_GAME_SHARE:.0%})")
    fill = (nonnull / rows).round(4) if rows else nonnull * 0.0
    print(f"  {year}: {len(pks)} games, {n_gap} with tracking gaps; fill pitch_type {fill['pitch_type']:.4f}, "
          f"spin_axis {fill['spin_axis']:.4f}, release_spin_rate {fill['release_spin_rate']:.4f}, arm_angle "
          f"{fill['arm_angle']:.4f} of rows / {arm_thrown / max(thrown_rows, 1):.4f} of thrown pitches; "
          f"{auto_rows} automatic balls/strikes", flush=True)
    summary = {"rows": rows, "games": len(pks), "key": key, "gaps": gaps, "fill": fill,
               "digest": hashlib.sha256(np.sort(np.concatenate(hashes)).tobytes()).hexdigest() if hashes else None}
    return summary, [f"{year}: {e}" for e in errors]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, required=True, help="dataset-metadata.json with {rows:<file>} placeholders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--days", help="comma-separated YYYY-MM-DD: a sample build (season-total gates per day instead)")
    ap.add_argument("--cache", type=Path, help="keep/reuse Savant's raw day files here (tests only)")
    args = ap.parse_args()

    only = {date.fromisoformat(x) for x in args.days.split(",")} if args.days else None
    full = only is None
    print(f"fetch/parse functions from {_D4.parent.name}/build.py md5 {md5(_D4)}", flush=True)
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    stage = args.out.with_name(args.out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    failures: list[str] = []
    try:
        summaries = {}
        for y in YEARS:
            summaries[y], errs = build_season(y, stage, only, args.cache, full)
            failures += errs
        got = {y for y in YEARS if summaries[y]["rows"]}
        if got != {2021, 2022, 2023}:
            failures.append(f"seasons with rows {sorted(got)} != [2021, 2022, 2023]")
        keys = np.concatenate([summaries[y]["key"] for y in YEARS])
        if len(np.unique(keys)) != len(keys):
            failures.append("duplicate pitch keys across seasons")
        digests = [summaries[y]["digest"] for y in YEARS if summaries[y]["digest"]]
        if len(set(digests)) != len(digests):
            failures.append("two seasons have the same content")
        for y in YEARS:
            s = pq.read_schema(stage / f"statcast_{y}.parquet").remove_metadata()
            if not s.equals(SCHEMA.remove_metadata()):
                failures.append(f"statcast_{y}.parquet schema {s} != the frozen schema {SCHEMA}")
        for y in YEARS:
            print(f"{y}: {summaries[y]['rows']:,} rows, {summaries[y]['games']} games")
        print("fill (share of rows non-null):\n" + pd.DataFrame({y: summaries[y]["fill"] for y in YEARS}).to_string())
        gaps = pd.DataFrame([g for y in YEARS for g in summaries[y]["gaps"]], columns=GAP_COLUMNS)
        if len(gaps):
            print(f"tracking gaps ({len(gaps)} games):\n" + gaps.to_string(index=False))
        if failures:
            print("\n".join(["", "Gates failed, nothing written:"] + failures))
            return 1

        meta = json.loads(args.meta.read_text(encoding="utf-8"))
        desc = meta["description"]
        for y in YEARS:
            desc = desc.replace("{rows:statcast_%d}" % y, f"{summaries[y]['rows']:,}")
        desc = desc.replace("{rows:tracking_gaps}", f"{len(gaps):,}")
        left = re.findall(r"\{rows:[a-z_0-9]+\}", desc)
        if left:
            print(f"description placeholders with no table: {left}")
            return 1
        meta["description"] = desc
        gaps.to_csv(stage / "tracking_gaps.csv", index=False)
        (stage / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(args.out, ignore_errors=True)
        stage.rename(args.out)
        print("md5 manifest (for the freeze file):")
        for y in YEARS:
            p = args.out / f"statcast_{y}.parquet"
            print(f"  {md5(p)}  {p.name}  {summaries[y]['rows']} rows  {p.stat().st_size} bytes")
        print(f"wrote {sorted(p.name for p in args.out.iterdir())} to {args.out}")
        return 0
    except d4.GateError as e:
        print(f"\nGate failed, nothing written: {e}")
        return 1
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
