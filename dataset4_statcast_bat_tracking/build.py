"""Build the Kaggle dataset "MLB Statcast + Bat Tracking (2024-2026)"
(yasunorim/mlb-statcast-bat-tracking-2024-2025; the slug is kept from the first version).

Run by `.github/workflows/update-dataset.yml` (via build.sh) before `kaggle datasets version`.

Sources
- Pitch-by-pitch data straight from Baseball Savant's statcast_search CSV endpoint, one request per
  calendar day of each regular season, regular-season games only (hfGT=R|). The first version was
  made with pybaseball's statcast(), whose request asks for game types R|PO|S: it carried 20,547
  spring-training pitches (2024-03-20..03-26, Savant's count today) and missed the 2025 Tokyo
  opener (03-18/19, 625 pitches) and 2024-09-30 (577 pitches).
- Season dates and the list of regular-season games from MLB StatsAPI (/seasons, /schedule).
- Player names, bats/throws from MLB StatsAPI /people (Savant's player_name is the pitcher).

Each season is fetched and written (Parquet, zstd) before the next one starts, so memory holds one
season's small aggregation columns at most. Everything is written to a staging folder that becomes
the output folder only if every gate passes; otherwise the staging folder is removed and nothing is
written.

Gates (the build fails and writes nothing if any is violated):
- every day's response: header is Savant's known column set, game_date is the requested day,
  game_type is R only, game_year is the season, fewer than DAY_MAX rows, known pitch descriptions,
  integer columns hold integers, zone in 1-9/11-14;
- on days with at least NULL_CHECK_MIN_ROWS pitches, no kept column is empty for the whole day
  (except days-since-previous-game on the season's first two days and days-until-next-game on its
  last two), which catches a day Savant has not finished backfilling;
- seasons are exactly YEARS; rows per season >= ROW_FLOOR, rows with a pitch_type equal to Savant's
  own season total (group_by=team, which leaves out pitch-clock automatic balls/strikes), and no kept column is empty for the whole season [full build only];
- the season's first and last regular-season day have rows;
- every regular-season game StatsAPI lists as Final on a fetched day is present, and no game outside
  the season's regular-season schedule is;
- no duplicate pitch_uid; no two seasons with the same content;
- 2026 bat_speed non-null share >= BAT_FLOOR_2026;
- the dropped columns are empty in every row, and the column set contains every column of the previous version except the dropped ones;
- every batter and pitcher id resolves in players.csv;
- every output file and column is described in settings.json, and nothing else is.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
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
import requests

YEARS = (2024, 2025, 2026)
SAVANT = "https://baseballsavant.mlb.com/statcast_search/csv"
STATSAPI = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "kaggle-datasets-build/1.0 (+https://github.com/yasumorishima/kaggle-datasets)"}
SLEEP = 1.0
RETRY_WAIT = (5, 15, 45, 90)

# Savant counted 709,512 (2024) and 710,084 (2025) regular-season pitches with a pitch type
# (measured 2026-09-28); the rows here also include the pitch-clock automatic balls/strikes.
ROW_FLOOR = 700_000
# bat_speed was non-null on 45.0-46.1% of pitches on the 2026 days measured (swings only).
BAT_FLOOR_2026 = 0.40
# A real day has at most ~5,500 pitches; anything near this means a truncated or wrong answer.
DAY_MAX = 25_000
NULL_CHECK_MIN_ROWS = 200

# Savant's statcast_search CSV header (type=details), as of 2026-09-28.
SAVANT_COLUMNS = (
    "pitch_type game_date release_speed release_pos_x release_pos_z player_name batter pitcher events "
    "description spin_dir spin_rate_deprecated break_angle_deprecated break_length_deprecated zone des "
    "game_type stand p_throws home_team away_team type hit_location bb_type balls strikes game_year pfx_x "
    "pfx_z plate_x plate_z on_3b on_2b on_1b outs_when_up inning inning_topbot hc_x hc_y tfs_deprecated "
    "tfs_zulu_deprecated umpire sv_id vx0 vy0 vz0 ax ay az sz_top sz_bot hit_distance_sc launch_speed "
    "launch_angle effective_speed release_spin_rate release_extension game_pk fielder_2 fielder_3 "
    "fielder_4 fielder_5 fielder_6 fielder_7 fielder_8 fielder_9 release_pos_y "
    "estimated_ba_using_speedangle estimated_woba_using_speedangle woba_value woba_denom babip_value "
    "iso_value launch_speed_angle at_bat_number pitch_number pitch_name home_score away_score bat_score "
    "fld_score post_away_score post_home_score post_bat_score post_fld_score if_fielding_alignment "
    "of_fielding_alignment spin_axis delta_home_win_exp delta_run_exp bat_speed swing_length "
    "miss_distance estimated_slg_using_speedangle delta_pitcher_run_exp hyper_speed home_score_diff "
    "bat_score_diff home_win_exp bat_win_exp age_pit_legacy age_bat_legacy age_pit age_bat "
    "n_thruorder_pitcher n_priorpa_thisgame_player_at_bat pitcher_days_since_prev_game "
    "batter_days_since_prev_game pitcher_days_until_next_game batter_days_until_next_game "
    "api_break_z_with_gravity api_break_x_arm api_break_x_batter_in arm_angle attack_angle "
    "attack_direction swing_path_tilt intercept_ball_minus_batter_pos_x_inches "
    "intercept_ball_minus_batter_pos_y_inches"
).split()
# Deprecated fields of the old tracking system; empty on every pitch of 2024-2026.
DROPPED_EMPTY = (
    "spin_dir", "spin_rate_deprecated", "break_angle_deprecated", "break_length_deprecated",
    "tfs_deprecated", "tfs_zulu_deprecated", "umpire", "sv_id",
)
STRING_COLS = {
    "pitch_type", "player_name", "events", "description", "des", "game_type", "stand", "p_throws",
    "home_team", "away_team", "type", "bb_type", "inning_topbot", "pitch_name",
    "if_fielding_alignment", "of_fielding_alignment", "spin_dir", "umpire", "sv_id",
    "tfs_deprecated", "tfs_zulu_deprecated",
}
INT_COLS = {
    "game_pk", "batter", "pitcher", "at_bat_number", "pitch_number", "inning", "balls", "strikes",
    "outs_when_up", "game_year", "zone", "on_1b", "on_2b", "on_3b", "fielder_2", "fielder_3",
    "fielder_4", "fielder_5", "fielder_6", "fielder_7", "fielder_8", "fielder_9", "home_score",
    "away_score", "bat_score", "fld_score", "post_away_score", "post_home_score", "post_bat_score",
    "post_fld_score", "home_score_diff", "bat_score_diff", "hit_location", "launch_speed_angle",
    "woba_denom", "babip_value", "iso_value", "age_pit", "age_bat", "age_pit_legacy", "age_bat_legacy",
    "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat", "pitcher_days_since_prev_game",
    "batter_days_since_prev_game", "pitcher_days_until_next_game", "batter_days_until_next_game",
}
DATE_COLS = {"game_date"}
SINCE_COLS = {"pitcher_days_since_prev_game", "batter_days_since_prev_game"}
UNTIL_COLS = {"pitcher_days_until_next_game", "batter_days_until_next_game"}

# Pitch descriptions. A swing is any pitch the batter offered at; a whiff is a swing that missed
# the ball. Foul tips touch the bat, so they are swings but not whiffs.
WHIFF = {"swinging_strike", "swinging_strike_blocked", "missed_bunt", "swinging_pitchout"}
CONTACT = {"foul", "foul_tip", "foul_bunt", "bunt_foul_tip", "foul_pitchout", "hit_into_play"}
TAKE = {"ball", "blocked_ball", "called_strike", "hit_by_pitch", "pitchout", "intent_ball",
        "automatic_ball", "automatic_strike"}
SWING = WHIFF | CONTACT
KNOWN_DESCRIPTIONS = SWING | TAKE
ZONES_IN = set(range(1, 10))
ZONES_OUT = {11, 12, 13, 14}

KEPT = [c for c in SAVANT_COLUMNS if c not in DROPPED_EMPTY]
DERIVED = ["pitch_uid", "is_swing", "is_whiff", "in_zone"]
PITCH_COLUMNS = KEPT + DERIVED


def _arrow_type(c: str) -> pa.DataType:
    if c in DATE_COLS:
        return pa.date32()
    if c in STRING_COLS or c == "pitch_uid":
        return pa.string()
    if c in INT_COLS:
        return pa.int64()
    if c in ("is_swing", "is_whiff", "in_zone"):
        return pa.bool_()
    return pa.float64()


SCHEMA = pa.schema([pa.field(c, _arrow_type(c)) for c in PITCH_COLUMNS])
# Columns kept in memory for the season tables.
SLIM = ["game_year", "batter", "pitcher", "description", "type", "events", "pitch_type",
        "release_speed", "bat_speed", "swing_length", "attack_angle", "attack_direction",
        "swing_path_tilt", "launch_speed", "estimated_woba_using_speedangle", "woba_value",
        "woba_denom", "is_swing", "is_whiff", "in_zone"]


class GateError(Exception):
    pass


def _get(url: str, params: dict | None = None, timeout: int = 300) -> requests.Response:
    last = None
    for wait in (*RETRY_WAIT, None):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
            if r.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {r.status_code}"
            else:
                r.raise_for_status()
                return r
        except (requests.ConnectionError, requests.Timeout) as e:
            last = f"{type(e).__name__}: {e}"
        if wait is None:
            break
        print(f"  retry in {wait}s after {last}: {url}", flush=True)
        time.sleep(wait)
    raise SystemExit(f"giving up on {url} {params or ''}: {last}")


def season_dates(year: int) -> tuple[date, date]:
    s = _get(f"{STATSAPI}/seasons/{year}", {"sportId": 1}, timeout=60).json()["seasons"][0]
    return date.fromisoformat(s["regularSeasonStartDate"]), date.fromisoformat(s["regularSeasonEndDate"])


def schedule(start: date, end: date) -> pd.DataFrame:
    js = _get(f"{STATSAPI}/schedule", {"sportId": 1, "gameType": "R", "startDate": start.isoformat(),
                                       "endDate": end.isoformat()}, timeout=120).json()
    rows = [{"game_pk": g["gamePk"], "official_date": g.get("officialDate") or d["date"],
             "state": g["status"].get("codedGameState")}
            for d in js.get("dates", []) for g in d.get("games", [])]
    if not rows:
        raise SystemExit(f"StatsAPI schedule {start}..{end}: no games")
    return pd.DataFrame(rows)


def savant_total(year: int) -> int:
    """Savant's own count of regular-season pitches for the season (one row per team)."""
    time.sleep(SLEEP)
    r = _get(SAVANT, {"all": "true", "hfGT": "R|", "hfSea": f"{year}|", "player_type": "pitcher",
                      "group_by": "team", "min_pitches": 0, "min_results": 0, "sort_col": "pitches",
                      "sort_order": "desc"})
    df = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
    if len(df) != 30 or "pitches" not in df:
        raise SystemExit(f"Savant season total {year}: unexpected answer {df.shape}")
    return int(df["pitches"].sum())


def fetch_day(day: date, cache: Path | None) -> str:
    path = cache / f"{day.isoformat()}.csv" if cache else None
    if path is not None and path.exists():
        return path.read_bytes().decode("utf-8-sig")
    params = {"all": "true", "hfGT": "R|", "hfSea": "", "player_type": "pitcher",
              "game_date_gt": day.isoformat(), "game_date_lt": day.isoformat(), "min_pitches": 0,
              "min_results": 0, "group_by": "name", "sort_col": "pitches", "sort_order": "desc",
              "type": "details"}
    for attempt in range(3):
        time.sleep(SLEEP)
        r = _get(SAVANT, params)
        text = r.content.decode("utf-8-sig")
        if text.lstrip().startswith('"pitch_type"') or text.lstrip().startswith("pitch_type"):
            break
        print(f"  {day}: answer is not the CSV ({text[:80]!r}), retrying", flush=True)
        time.sleep(RETRY_WAIT[min(attempt, len(RETRY_WAIT) - 1)])
    else:
        raise SystemExit(f"{day}: Savant did not answer with the CSV")
    if path is not None:
        path.write_bytes(r.content)
    return text


def parse_day(text: str, day: date, year: int) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), dtype={c: "string" for c in STRING_COLS | DATE_COLS}, low_memory=False)
    if set(df.columns) != set(SAVANT_COLUMNS) or len(df.columns) != len(SAVANT_COLUMNS):
        raise GateError(f"{day}: columns changed: new {sorted(set(df.columns) - set(SAVANT_COLUMNS))}, "
                        f"missing {sorted(set(SAVANT_COLUMNS) - set(df.columns))}")
    if len(df) >= DAY_MAX:
        raise GateError(f"{day}: {len(df)} rows, a truncated or wrong answer")
    if len(df) == 0:
        return df
    bad = []
    if (df["game_date"] != day.isoformat()).any():
        bad.append(f"game_date {sorted(df['game_date'].dropna().unique())[:5]}")
    if set(df["game_type"].dropna()) != {"R"} or df["game_type"].isna().any():
        bad.append(f"game_type {sorted(df['game_type'].dropna().unique())}")
    for c in SAVANT_COLUMNS:
        if c in STRING_COLS or c in DATE_COLS:
            continue
        s = pd.to_numeric(df[c], errors="raise")
        if c in INT_COLS:
            v = s.dropna()
            if len(v) and not np.all(np.equal(np.mod(v.to_numpy(dtype="float64"), 1), 0)):
                bad.append(f"{c} has non-integer values")
                continue
            df[c] = s.astype("Int64")
        else:
            df[c] = s.astype("float64")
    if (df["game_year"] != year).any():
        bad.append(f"game_year {sorted(df['game_year'].dropna().unique())}")
    unknown = set(df["description"].dropna()) - KNOWN_DESCRIPTIONS
    if unknown or df["description"].isna().any():
        bad.append(f"descriptions not classified: {sorted(unknown)} (null {int(df['description'].isna().sum())})")
    zones = set(df["zone"].dropna().astype(int))
    if zones - ZONES_IN - ZONES_OUT:
        bad.append(f"zones {sorted(zones - ZONES_IN - ZONES_OUT)}")
    if bad:
        raise GateError(f"{day}: " + "; ".join(bad))
    df["game_date"] = pd.to_datetime(df["game_date"], format="%Y-%m-%d").dt.date
    df["pitch_uid"] = (df["game_pk"].astype(str) + "-" + df["at_bat_number"].astype(str) + "-"
                       + df["pitch_number"].astype(str))
    df["is_swing"] = df["description"].isin(SWING).astype(bool)
    df["is_whiff"] = df["description"].isin(WHIFF).astype(bool)
    z = df["zone"]
    df["in_zone"] = pd.array([pd.NA if pd.isna(v) else (int(v) in ZONES_IN) for v in z], dtype="boolean")
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)


def _key(df: pd.DataFrame) -> np.ndarray:
    ab = df["at_bat_number"].to_numpy(dtype="int64")
    pn = df["pitch_number"].to_numpy(dtype="int64")
    if len(ab) and (ab.max() >= 1000 or pn.max() >= 1000):
        raise GateError("at_bat_number or pitch_number >= 1000")
    return df["game_pk"].to_numpy(dtype="int64") * 1_000_000 + ab * 1000 + pn


def _content_digest(slim: pd.DataFrame) -> str:
    """Content without the season-naming columns: the same pitches under another year would match."""
    body = slim[["batter", "pitcher", "description", "release_speed", "bat_speed", "launch_speed"]]
    h = pd.util.hash_pandas_object(body, index=False).to_numpy()
    return hashlib.sha256(np.sort(h).tobytes()).hexdigest()


def build_season(year: int, stage: Path, only_days: set[date] | None, cache: Path | None, full: bool):
    """Fetch, check and write one season. Returns (slim frame, summary dict, errors)."""
    errors: list[str] = []
    start, end = season_dates(year)
    sched = schedule(start, end)
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    game_days = sorted({date.fromisoformat(d) for d in sched["official_date"]})
    first_days, last_days = set(game_days[:2]), set(game_days[-2:])
    if only_days is not None:
        days = [d for d in days if d in only_days]
    print(f"== {year}: regular season {start}..{end}, {sched['game_pk'].nunique()} games scheduled, "
          f"{len(days)} days to fetch", flush=True)
    writer = pq.ParquetWriter(stage / f"statcast_{year}.parquet", SCHEMA, compression="zstd")
    buf: list[pd.DataFrame] = []
    buffered = 0
    slims, keys, pks = [], [], set()
    nonnull = pd.Series(0, index=SAVANT_COLUMNS, dtype="int64")
    rows = 0
    per_day = {}
    t0 = time.time()
    try:
        for d in days:
            t = time.time()
            df = parse_day(fetch_day(d, cache), d, year)
            per_day[d] = len(df)
            print(f"  {d} {len(df):>5} rows {time.time() - t:5.1f}s", flush=True)
            if not len(df):
                continue
            nonnull += df[SAVANT_COLUMNS].notna().sum()
            if len(df) >= NULL_CHECK_MIN_ROWS:
                allowed = set(DROPPED_EMPTY) | (SINCE_COLS if d in first_days else set()) | (
                    UNTIL_COLS if d in last_days else set())
                empty = [c for c in SAVANT_COLUMNS if c not in allowed and df[c].isna().all()]
                if empty:
                    errors.append(f"{d}: {len(df)} pitches but empty {empty} (not backfilled yet?)")
            keys.append(_key(df))
            pks.update(df["game_pk"].astype(int).unique().tolist())
            slims.append(df[SLIM].copy())
            rows += len(df)
            buf.append(df[PITCH_COLUMNS])
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

    slim = pd.concat(slims, ignore_index=True) if slims else pd.DataFrame(columns=SLIM)
    key = np.concatenate(keys) if keys else np.array([], dtype="int64")
    fetched = set(days)
    if full and rows < ROW_FLOOR:
        errors.append(f"{rows:,} rows < {ROW_FLOOR:,}")
    if full:
        # Savant's grouped count leaves out pitches with no pitch_type (automatic balls/strikes of
        # the pitch clock): it matched the rows with a pitch_type exactly on every day checked.
        total = savant_total(year)
        typed = int(slim["pitch_type"].notna().sum())
        if typed != total:
            errors.append(f"{typed:,} rows with a pitch_type but Savant counts {total:,} regular-season pitches")
    for d in (start, end):
        if d in fetched and per_day.get(d, 0) == 0:
            errors.append(f"regular-season day {d} has no pitches")
    final = sched[(sched["state"] == "F") & sched["official_date"].isin({x.isoformat() for x in fetched})]
    missing = sorted(set(final["game_pk"]) - pks)
    if missing:
        errors.append(f"{len(missing)} Final regular-season games with no pitches, e.g. {missing[:5]}")
    outside = sorted(pks - set(sched["game_pk"]))
    if outside:
        errors.append(f"games not in the regular-season schedule: {outside[:5]}")
    if len(np.unique(key)) != len(key):
        errors.append(f"{len(key) - len(np.unique(key))} duplicate pitch_uid")
    for c in DROPPED_EMPTY:
        if nonnull[c]:
            errors.append(f"dropped column {c} has {int(nonnull[c])} values")
    # A whole-season property: a sample of a few days can legitimately miss a column.
    empty_all = [c for c in KEPT if nonnull[c] == 0]
    if full and empty_all:
        errors.append(f"columns empty in every row of the season: {empty_all}")
    share = float(slim["bat_speed"].notna().mean()) if rows else 0.0
    summary = {"rows": rows, "games": len(pks), "bat_speed_share": share, "key": key,
               "digest": _content_digest(slim) if rows else None,
               "first_day": per_day.get(start), "last_day": per_day.get(end)}
    return slim, summary, [f"{year}: {e}" for e in errors]


def _rate(num, den):
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def batter_table(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["pa"] = s["events"].notna()
    s["bip"] = s["type"] == "X"
    s["ev"] = s["launch_speed"].where(s["bip"])
    s["hard"] = s["bip"] & (s["launch_speed"] >= 95)
    s["xw"] = s["estimated_woba_using_speedangle"].where(s["bip"])
    s["zsw"] = s["is_swing"] & (s["in_zone"] == True)  # noqa: E712
    s["osw"] = s["is_swing"] & (s["in_zone"] == False)  # noqa: E712
    s["zin"] = s["in_zone"] == True  # noqa: E712
    s["zout"] = s["in_zone"] == False  # noqa: E712
    # Savant's average bat speed is over 'competitive' swings: the fastest 90% of a player's swings,
    # plus any 60+ mph swing that produced a 90+ mph exit velocity.
    t = s[s["bat_speed"].notna()].copy()
    g = t.groupby(["batter", "game_year"])["bat_speed"]
    rank = g.rank(ascending=False, method="first")
    n = g.transform("size")
    t["competitive"] = (rank <= np.ceil(0.9 * n)) | ((t["bat_speed"] >= 60) & (t["launch_speed"] >= 90))
    comp = t[t["competitive"]].groupby(["batter", "game_year"]).agg(
        competitive_swings=("bat_speed", "size"), avg_bat_speed_competitive=("bat_speed", "mean"))
    out = s.groupby(["batter", "game_year"]).agg(
        pitches=("description", "size"), pa=("pa", "sum"), swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"), zone_pitches=("zin", "sum"), zone_swings=("zsw", "sum"),
        chase_pitches=("zout", "sum"), chase_swings=("osw", "sum"),
        tracked_swings=("bat_speed", "count"), avg_bat_speed=("bat_speed", "mean"),
        avg_swing_length=("swing_length", "mean"), avg_attack_angle=("attack_angle", "mean"),
        avg_attack_direction=("attack_direction", "mean"), avg_swing_path_tilt=("swing_path_tilt", "mean"),
        batted_balls=("bip", "sum"), avg_exit_velocity=("ev", "mean"), hard_hit=("hard", "sum"),
        xwoba_on_contact=("xw", "mean"), woba_value=("woba_value", "sum"), woba_denom=("woba_denom", "sum"),
    ).join(comp).reset_index()
    out["competitive_swings"] = out["competitive_swings"].fillna(0).astype("int64")
    return _finish(out, "batter")


def pitcher_table(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["pa"] = s["events"].notna()
    s["bip"] = s["type"] == "X"
    s["ev"] = s["launch_speed"].where(s["bip"])
    s["hard"] = s["bip"] & (s["launch_speed"] >= 95)
    s["xw"] = s["estimated_woba_using_speedangle"].where(s["bip"])
    s["cs"] = s["description"] == "called_strike"
    s["zsw"] = s["is_swing"] & (s["in_zone"] == True)  # noqa: E712
    s["osw"] = s["is_swing"] & (s["in_zone"] == False)  # noqa: E712
    s["zin"] = s["in_zone"] == True  # noqa: E712
    s["zout"] = s["in_zone"] == False  # noqa: E712
    s["ff"] = s["release_speed"].where(s["pitch_type"] == "FF")
    out = s.groupby(["pitcher", "game_year"]).agg(
        pitches=("description", "size"), batters_faced=("pa", "sum"), swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"), called_strikes=("cs", "sum"), zone_pitches=("zin", "sum"),
        zone_swings=("zsw", "sum"), chase_pitches=("zout", "sum"), chase_swings=("osw", "sum"),
        four_seam_pitches=("ff", "count"), avg_four_seam_velocity=("ff", "mean"),
        tracked_swings_against=("bat_speed", "count"), avg_bat_speed_against=("bat_speed", "mean"),
        batted_balls=("bip", "sum"), avg_exit_velocity=("ev", "mean"), hard_hit=("hard", "sum"),
        xwoba_on_contact=("xw", "mean"), woba_value=("woba_value", "sum"), woba_denom=("woba_denom", "sum"),
    ).reset_index()
    out["csw_rate"] = _rate(out["called_strikes"] + out["whiffs"], out["pitches"])
    return _finish(out, "pitcher")


def _finish(out: pd.DataFrame, who: str) -> pd.DataFrame:
    out["whiff_rate"] = _rate(out["whiffs"], out["swings"])
    out["zone_swing_rate"] = _rate(out["zone_swings"], out["zone_pitches"])
    out["chase_rate"] = _rate(out["chase_swings"], out["chase_pitches"])
    out["zone_rate"] = _rate(out["zone_pitches"], out["zone_pitches"] + out["chase_pitches"])
    out["hard_hit_rate"] = _rate(out["hard_hit"], out["batted_balls"])
    out["woba"] = _rate(out["woba_value"], out["woba_denom"])
    out = out.drop(columns=["woba_value", "woba_denom", "zone_swings", "chase_swings"])
    for c in out.columns:
        if out[c].dtype.kind == "b":
            out[c] = out[c].astype("int64")
        if out[c].dtype.kind == "f" and c not in (who,):
            out[c] = out[c].round(4)
    return out.sort_values(["game_year", who]).reset_index(drop=True)


def players_table(ids: list[int]) -> pd.DataFrame:
    rows = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        time.sleep(0.3)
        js = _get(f"{STATSAPI}/people", {"personIds": ",".join(map(str, chunk))}, timeout=120).json()
        for p in js.get("people", []):
            rows.append({
                "player_id": p["id"], "full_name": p.get("fullName"), "first_name": p.get("firstName"),
                "last_name": p.get("lastName"), "bats": (p.get("batSide") or {}).get("code"),
                "throws": (p.get("pitchHand") or {}).get("code"),
                "primary_position": (p.get("primaryPosition") or {}).get("abbreviation"),
                "birth_date": p.get("birthDate"), "mlb_debut_date": p.get("mlbDebutDate"),
            })
    return pd.DataFrame(rows).drop_duplicates("player_id").sort_values("player_id").reset_index(drop=True)


def previous_columns(md: Path) -> list[str]:
    return re.findall(r"^### \*\*(.+?)\*\*", md.read_text(encoding="utf-8"), re.M)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, required=True, help="dataset-metadata.json with {rows:<file>} placeholders")
    ap.add_argument("--settings", type=Path, required=True, help="settings.json (file and column descriptions)")
    ap.add_argument("--prev-columns", type=Path, required=True, help="dataset4_columns.md: the previous version's columns")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--days", help="comma-separated YYYY-MM-DD: a sample build (season-total gates skipped)")
    ap.add_argument("--cache", type=Path, help="keep/reuse Savant's raw day files here (tests only)")
    args = ap.parse_args()

    only = {date.fromisoformat(x) for x in args.days.split(",")} if args.days else None
    full = only is None
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
    stage = args.out.with_name(args.out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    failures: list[str] = []
    try:
        slims, summaries = {}, {}
        for y in YEARS:
            slims[y], summaries[y], errs = build_season(y, stage, only, args.cache, full)
            failures += errs

        got = {y for y in YEARS if summaries[y]["rows"]}
        if got != set(YEARS):
            failures.append(f"seasons with rows {sorted(got)} != {list(YEARS)}")
        keys = np.concatenate([summaries[y]["key"] for y in YEARS])
        if len(np.unique(keys)) != len(keys):
            failures.append("duplicate pitch_uid across seasons")
        digests = [summaries[y]["digest"] for y in YEARS if summaries[y]["digest"]]
        if len(set(digests)) != len(digests):
            failures.append("two seasons have the same content")
        share = summaries[2026]["bat_speed_share"]
        if share < BAT_FLOOR_2026:
            failures.append(f"2026 bat_speed non-null share {share:.3f} < {BAT_FLOOR_2026}")
        prev = previous_columns(args.prev_columns)
        if len(prev) != 118:
            failures.append(f"{args.prev_columns.name}: {len(prev)} columns, expected the previous version's 118")
        lost = sorted(set(prev) - set(DROPPED_EMPTY) - set(PITCH_COLUMNS))
        if lost:
            failures.append(f"previous columns missing from the new files: {lost}")
        schemas = {str(pq.read_schema(stage / f"statcast_{y}.parquet")) for y in YEARS}
        if len(schemas) != 1:
            failures.append("the season Parquet files have different schemas")

        slim = pd.concat([slims[y] for y in YEARS], ignore_index=True)
        del slims
        batters = batter_table(slim)
        pitchers = pitcher_table(slim)
        for name, t, who in (("batter_season", batters, "batter"), ("pitcher_season", pitchers, "pitcher")):
            per = t.groupby("game_year")["pitches"].sum()
            for y in YEARS:
                if int(per.get(y, 0)) != summaries[y]["rows"]:
                    failures.append(f"{name}: {y} pitches {int(per.get(y, 0)):,} != {summaries[y]['rows']:,} rows")
        ids = sorted(set(slim["batter"].dropna().astype(int)) | set(slim["pitcher"].dropna().astype(int)))
        players = players_table(ids)
        unresolved = sorted(set(ids) - set(players["player_id"]))
        if unresolved:
            failures.append(f"{len(unresolved)} batter/pitcher ids not in StatsAPI /people, e.g. {unresolved[:5]}")
        names = players.set_index("player_id")["full_name"]
        batters.insert(1, "batter_name", batters["batter"].map(names))
        pitchers.insert(1, "pitcher_name", pitchers["pitcher"].map(names))
        if players["full_name"].isna().any():
            failures.append("players without a name")

        tables = {"batter_season.csv": batters, "pitcher_season.csv": pitchers, "players.csv": players}
        columns = {f"statcast_{y}.parquet": PITCH_COLUMNS for y in YEARS}
        columns.update({k: list(v.columns) for k, v in tables.items()})
        settings = json.loads(args.settings.read_text(encoding="utf-8"))
        if set(settings["files"]) != set(columns):
            failures.append(f"settings.json files {sorted(settings['files'])} != built {sorted(columns)}")
        for f, cols in columns.items():
            s = settings["files"].get(f, {})
            if not s.get("description"):
                failures.append(f"settings.json: {f} has no description")
            described = s.get("columns", {})
            if set(described) != set(cols):
                failures.append(f"settings.json {f}: undescribed {sorted(set(cols) - set(described))}, "
                                f"not in the file {sorted(set(described) - set(cols))}")
            if any(not v for v in described.values()):
                failures.append(f"settings.json {f}: empty column descriptions")

        for y in YEARS:
            s = summaries[y]
            print(f"{y}: {s['rows']:,} rows, {s['games']} games, bat_speed share {s['bat_speed_share']:.3f}, "
                  f"first day {s['first_day']}, last day {s['last_day']}")
        print(f"batter_season {batters.shape}, pitcher_season {pitchers.shape}, players {players.shape}")
        if failures:
            print("\n".join(["", "Gates failed, nothing written:"] + failures))
            return 1

        meta = json.loads(args.meta.read_text(encoding="utf-8"))
        desc = meta["description"]
        for y in YEARS:
            desc = desc.replace("{rows:statcast_%d}" % y, f"{summaries[y]['rows']:,}")
        for name, t in (("batter_season", batters), ("pitcher_season", pitchers)):
            per = t["game_year"].value_counts()
            desc = desc.replace("{rows:" + name + "}", " / ".join(f"{int(per.get(y, 0)):,}" for y in YEARS))
        desc = desc.replace("{rows:players}", f"{len(players):,}")
        left = re.findall(r"\{rows:[a-z_0-9]+\}", desc)
        if left:
            print(f"description placeholders with no table: {left}")
            return 1
        missing = [f for f in columns if f"`{f}`" not in desc]
        if missing:
            print(f"files not in the description: {missing}")
            return 1
        meta["description"] = desc

        for f, t in tables.items():
            t.to_csv(stage / f, index=False)
        (stage / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(args.out, ignore_errors=True)
        stage.rename(args.out)
        for p in sorted(args.out.iterdir()):
            print(f"  {p.name:<24} {p.stat().st_size / 1e6:9.1f} MB")
        print(f"wrote {len(columns)} files and dataset-metadata.json to {args.out}")
        return 0
    except GateError as e:
        print(f"\nGate failed, nothing written: {e}")
        return 1
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
