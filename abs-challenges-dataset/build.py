"""Build the Kaggle dataset "ABS Challenges: Triple-A 2025 to MLB 2026".

Run by `.github/workflows/update-dataset.yml` (via build.sh). Nothing here is run locally.

Source: Baseball Savant's ABS challenge leaderboard, read through savant-extras 0.6.0
`abs_challenges()`, which parses the JSON the page embeds (the page's CSV export has no player id)
and raises if Savant answers with a season or level other than the one asked for.

Every (level, season, game type, challenger) board is fetched with min_challenges=0, so players
who had challenge opportunities but never challenged are listed too. enrich.py then adds per-player
columns (bio, season stats, Statcast aggregates) for each board's own level, season and game type.
The build fails, writing nothing, if any gate below or in enrich.py is violated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import savant_extras as sx
from savant_extras._http import EmptySavantResponse

import enrich

CHALLENGERS = ("batter", "pitcher", "catcher")
# (level, season, game type) boards that must have rows. MLB tested the system in 2025 spring
# training. Boards measured empty (2026-09-27): MLB 2025 regular season, MLB 2024 spring training,
# Triple-A spring training 2025 and 2026. The description says the MLB 2025 regular season and
# Triple-A spring training have no boards, so those are fetched and must stay empty; if Savant
# starts serving them the build fails.
EXPECTED = (("mlb", 2025, "S"), ("mlb", 2026, "S"), ("mlb", 2026, "R"),
            ("aaa", 2025, "R"), ("aaa", 2026, "R"))
MUST_BE_EMPTY = (("mlb", 2025, "R"), ("aaa", 2025, "S"), ("aaa", 2026, "S"))
# Minimum rows per board with min_challenges=0. Catchers are the smallest group (~110 in MLB).
MIN_ROWS = {"batter": 300, "pitcher": 300, "catcher": 80}
# Boards whose challengers must have made challenges (pitchers barely challenged in AAA 2025).
MIN_CHALLENGES = {"batter": 500, "catcher": 500}
BRIDGE_MIN = {"batter": 150, "catcher": 40}
# Largest share of rows two different boards may have in common on (player_id, n_total_sample,
# n_challenges, n_overturns). Measured on real boards: 0.01 at most. A board Savant served in
# place of another (a parameter ignored) shares nearly all of its rows even if a live game moved
# a few counts between the two requests, which an exact digest comparison would miss.
MAX_SHARED = 0.2
SHARED_KEY = ["player_id", "n_total_sample", "n_challenges", "n_overturns"]
SLEEP = 1.0
RETRIES = 3

# Bio columns from enrich.py, carried into the bridge table once (they do not depend on the season).
BIO_COLS = ("bats", "throws", "birth_date", "height_in", "weight_lb", "primary_position",
            "mlb_debut_date", "birth_country")
# enrich.py columns carried into the bridge table from each side (batters and catchers only).
ENRICH_PREFIXES = ("api_bat_", "api_c_", "sc_bat_", "sc_c_")
# Columns carried into the bridge table, from each side.
BRIDGE_COLS = (
    "team_abbr", "parent_org", "n_total_sample", "n_challenges", "n_overturns", "n_fails",
    "rate_challenges", "rate_overturns", "exp_chal", "exp_rate_overturns", "overturns_vs_exp",
    "n_chal_runs", "exp_chal_runs", "runs_vs_exp", "n_chal_reasonable_opps",
    "rate_reasonable_opp_taken", "n_strikeouts", "n_walks",
)


def _fetch(level: str, year: int, game_type: str, challenger: str) -> pd.DataFrame:
    for attempt in range(RETRIES):
        time.sleep(SLEEP * (1 + 4 * attempt))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EmptySavantResponse)
                df = sx.abs_challenges(year, level=level, challenge_type=challenger,
                                       game_type=game_type, min_challenges=0)
            break
        except requests.RequestException as e:
            if attempt == RETRIES - 1:
                raise
            print(f"retry {level} {year} {game_type} {challenger}: {e}")
    if df.empty:
        return df
    df.insert(df.columns.get_loc("year") + 1, "game_type", game_type)
    # `year_1` repeats `year`; drop it only when it does.
    if "year_1" in df.columns and (df["year_1"] == df["year"]).all():
        df = df.drop(columns="year_1")
    return df


def _shared(a: pd.DataFrame, b: pd.DataFrame) -> float:
    ka = set(map(tuple, a[SHARED_KEY].values))
    kb = set(map(tuple, b[SHARED_KEY].values))
    return len(ka & kb) / min(len(ka), len(kb))


def check_board(key: tuple, df: pd.DataFrame) -> list[str]:
    level, year, game_type, challenger = key
    errs = []
    if df["player_id"].duplicated().any():
        errs.append(f"{int(df['player_id'].duplicated().sum())} duplicated player_id")
    if not (df["n_challenges"] == 0).any():
        errs.append("no player with 0 challenges (min_challenges=0 was not applied)")
    if ("fielder_2" in df.columns) != (challenger == "catcher"):
        errs.append("fielder_2 column present on a non-catcher board or missing on a catcher board")
    if len(df) < MIN_ROWS[challenger]:
        errs.append(f"{len(df)} rows < {MIN_ROWS[challenger]}")
    if game_type == "R" and challenger in MIN_CHALLENGES and df["n_challenges"].sum() < MIN_CHALLENGES[challenger]:
        errs.append(f"{int(df['n_challenges'].sum())} challenges < {MIN_CHALLENGES[challenger]}")
    # Savant's own columns must agree with each other where they are defined from counts.
    if not (df["n_fails"] == df["n_challenges"] - df["n_overturns"]).all():
        errs.append("n_fails != n_challenges - n_overturns")
    m = df["n_total_sample"] > 0
    if not np.allclose(df.loc[m, "rate_challenges"], df.loc[m, "n_challenges"] / df.loc[m, "n_total_sample"], atol=1e-9):
        errs.append("rate_challenges != n_challenges / n_total_sample")
    m = df["n_challenges"] > 0
    if not np.allclose(df.loc[m, "rate_overturns"], df.loc[m, "n_overturns"] / df.loc[m, "n_challenges"], atol=1e-9):
        errs.append("rate_overturns != n_overturns / n_challenges")
    return errs


def _side_cols(df: pd.DataFrame) -> list[str]:
    return [*BRIDGE_COLS, "age", *[c for c in df.columns if c.startswith(ENRICH_PREFIXES)]]


def build_bridge(boards: dict) -> pd.DataFrame:
    frames = []
    for challenger in ("batter", "catcher"):
        a = boards[("aaa", 2025, "R", challenger)]
        b = boards[("mlb", 2026, "R", challenger)]
        a = a[a["n_total_sample"] > 0]
        b = b[b["n_total_sample"] > 0]
        ca, cb = _side_cols(a), _side_cols(b)
        left = a[["player_id", "player_name", *ca]].rename(columns={c: f"{c}_aaa2025" for c in ca})
        right = b[["player_id", *BIO_COLS, *cb]].rename(columns={c: f"{c}_mlb2026" for c in cb})
        j = left.merge(right, on="player_id", how="inner", validate="one_to_one")
        j.insert(0, "challenge_type", challenger)
        frames.append(j)
    out = pd.concat(frames, ignore_index=True)
    # Order: identity, bio, then each side's columns in the players table's order.
    ref = [*BRIDGE_COLS, "age"] + [c for c in pd.concat(list(boards.values())).columns if c.startswith(ENRICH_PREFIXES)]
    ref = list(dict.fromkeys(ref))
    side = [f"{c}_{sfx}" for sfx in ("aaa2025", "mlb2026") for c in ref if f"{c}_{sfx}" in out.columns]
    order = ["challenge_type", "player_id", "player_name", *BIO_COLS, *side]
    assert set(order) == set(out.columns), sorted(set(out.columns) ^ set(order))
    return out[order]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, required=True, help="dataset-metadata.json with {rows:...} placeholders")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(f"savant-extras {sx.__version__}")

    boards: dict = {}
    failures = []
    for level, year, game_type in EXPECTED + MUST_BE_EMPTY:
        for challenger in CHALLENGERS:
            key = (level, year, game_type, challenger)
            df = _fetch(level, year, game_type, challenger)
            required = (level, year, game_type) in EXPECTED
            if df.empty != (not required):
                print(f"FAIL {key} {'empty' if df.empty else f'{len(df)} rows, expected none'}")
                failures.append(f"{key}: {'empty' if df.empty else 'expected no rows'}")
                continue
            if df.empty:
                print(f"ok   {key} empty, as expected")
                continue
            errs = check_board(key, df)
            print(f"{'FAIL' if errs else 'ok  '} {key} rows={len(df)} challenges={int(df['n_challenges'].sum())}")
            failures += [f"{key}: {e}" for e in errs]
            boards[key] = df

    # No two boards, of any challenger, may be largely the same rows (the failure mode of Savant
    # ignoring a parameter and answering with a default board).
    keys = list(boards)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            share = _shared(boards[a], boards[b])
            if share > MAX_SHARED:
                failures.append(f"{a} and {b} share {share:.0%} of their rows")

    if failures:
        print("\n".join(["", "Gates failed, nothing written:"] + failures))
        return 1

    boards, errs = enrich.enrich(boards)
    if errs:
        print("\n".join(["", "Enrichment gates failed, nothing written:"] + errs))
        return 1

    players = pd.concat(boards.values(), ignore_index=True)
    bridge = build_bridge(boards)
    n_bridge = {c: int((bridge["challenge_type"] == c).sum()) for c in BRIDGE_MIN}
    print(f"bridge: {n_bridge}")
    short = {c: n for c, n in n_bridge.items() if n < BRIDGE_MIN[c]}
    if short:
        print(f"Gate failed, nothing written: bridge has {short} < {BRIDGE_MIN}")
        return 1

    files = {
        "abs_challenges_players.csv": players,
        "abs_aaa2025_to_mlb2026.csv": bridge,
    }
    counts = {"players": len(players), "bridge_batters": n_bridge["batter"],
              "bridge_catchers": n_bridge["catcher"], "columns": players.shape[1],
              "bridge_columns": bridge.shape[1]}
    for key, df in boards.items():
        level, year, game_type, challenger = key
        counts[f"{level}{year}{game_type}_{challenger}"] = len(df)
        counts[f"{level}{year}{game_type}_{challenger}_chal"] = int(df["n_challenges"].sum())

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    desc = meta["description"].replace("{built}", dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))
    for name, n in counts.items():
        desc = desc.replace("{rows:" + name + "}", f"{n:,}")
    left = re.findall(r"\{rows:[A-Za-z0-9_]+\}", desc)
    if left:
        print(f"description placeholders with no count: {left}")
        return 1
    meta["description"] = desc

    args.out.mkdir(parents=True, exist_ok=True)
    for name, df in files.items():
        df.to_csv(args.out / name, index=False)
        print(f"{name}: {df.shape}")
    (args.out / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(files)} CSVs and dataset-metadata.json to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
