"""Build the Kaggle dataset "Japanese MLB Players Statcast" (yasunorim/japan-mlb-pitchers-batters-statcast).

Run by `.github/workflows/update-dataset.yml` (via build.sh) before the new version is uploaded.

Why this exists: the previous version was assembled by hand once (February 2026) and no script
was kept. Its players.csv gave Hisashi Iwakuma the id 461325 (Tyler Clippard) and Yuki Matsui the
id 680686 (Josiah Gray), so those rows were other pitchers' pitches; several player-seasons were
missing (Ohtani pitching 2025, Darvish 2016, Hirano 2020, Arihara 2022, Kawasaki 2016); and spring
training and postseason pitches were mixed in with the regular season.

What it does
- Players are generated, not typed: every MLB StatsAPI season roster (/sports/1/players) from
  FIRST to LAST, birthCountry == "Japan", plus the explicit HERITAGE list (US-born, Japanese
  heritage, kept from the previous version). A player-season-role is eligible when StatsAPI's
  regular-season stats show pitches thrown (pitcher) or plate appearances (batter). Pitchers who
  batted are in the batting file.
- Pitches: one Baseball Savant statcast_search CSV per eligible player, season and role, regular
  season only (hfGT=R|), kept exactly as Savant writes it (read as text) plus player_name_eng.
- season_summary.csv: StatsAPI season and sabermetrics stats per player-season-role.

Gates (the build fails and writes nothing if any is violated): every eligible player-season-role
has Statcast rows; every row's season, game type (R only) and player id are the requested ones and
Savant's player_name matches StatsAPI's name for that id; the number of Statcast pitches per
player-season equals StatsAPI's pitch count (see NOT_PITCHES); no duplicate (game_pk, at_bat_number, pitch_number)
in a file; no game appears in two seasons of the same player and role; every Savant response has
the same columns; the columns the description calls empty are empty and hyper_speed follows the
rule its description states; every id of the previous
players.csv is kept or listed in CORRECTED; the player-seasons the description says were restored
are present; every output column has a description in settings.json.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

FIRST, LAST = 2015, 2026
# US-born players of Japanese heritage who were in the previous version (ids checked on StatsAPI:
# 663457 Lars Nootbaar, 641741 Gosuke Katoh, both birthCountry USA).
HERITAGE = {663457: "Lars Nootbaar", 641741: "Gosuke Katoh"}
# mlbam_id of the 34 rows of version 1 (the hand-built February 2026 players.csv, downloaded by
# version number). in_previous_version refers to this version; the carried players.csv is the
# latest version, which after the first rebuild is this build's own output.
VERSION1_IDS = frozenset({400085, 461325, 493114, 493117, 493128, 493157, 493159, 506433, 538506,
                          547749, 547888, 579328, 608372, 617228, 628317, 628318, 641741, 660261,
                          660271, 660294, 663457, 673451, 673540, 673548, 673633, 680686, 683822,
                          684007, 685493, 685503, 807799, 808963, 808967, 829272})
# Wrong ids in the previous players.csv -> the player's real id.
CORRECTED = {461325: 547874, 680686: 673513}
CORRECTED_NOTE = {461325: "Tyler Clippard", 680686: "Josiah Gray"}
# Player-season-roles missing from the previous version; the description says they are now included.
RESTORED = ((660271, 2025, "pitcher"), (506433, 2016, "pitcher"), (673633, 2020, "pitcher"),
            (685503, 2022, "pitcher"), (493128, 2016, "batter"))
# Deprecated Savant columns the column descriptions call empty in every row.
ALWAYS_EMPTY = ("umpire", "spin_dir", "spin_rate_deprecated", "break_angle_deprecated",
                "break_length_deprecated", "tfs_deprecated", "tfs_zulu_deprecated")
# Savant also writes rows that are not pitches: automatic balls and strikes (an automatic
# intentional walk, since 2017, is usually four automatic_ball rows; pitch-clock violations since 2023).
# StatsAPI's numberOfPitches does not count them. Measured on all 153 player-season-roles
# 2015-2026 (2026-09-28): Savant's rows minus these equal StatsAPI's pitch count exactly in every
# player-season whose games Savant had loaded; the only difference was a game played the day
# before, not yet in Savant. So the count must match exactly: a short count means Savant has not
# loaded every game yet (build again later), any other difference means the wrong pitches.
NOT_PITCHES = ("automatic_ball", "automatic_strike")
MAX_PITCH_DIFF = 0
SAVANT_SLEEP = 1.0
API = "https://statsapi.mlb.com/api/v1"
SAVANT = "https://baseballsavant.mlb.com/statcast_search/csv"
KEY = ("game_pk", "at_bat_number", "pitch_number")
# role -> (StatsAPI group, Savant id column, Savant lookup parameter, output file)
ROLES = {
    "pitcher": ("pitching", "pitcher", "pitchers_lookup[]", "japanese_mlb_pitching.csv"),
    "batter": ("hitting", "batter", "batters_lookup[]", "japanese_mlb_batting.csv"),
}
PLAYERS_FILE = "players.csv"
SUMMARY_FILE = "season_summary.csv"

_session = requests.Session()
_session.headers["User-Agent"] = "kaggle-datasets-build/1.0 (github.com/yasumorishima/kaggle-datasets)"


def get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response:
    err = ""
    for i in range(tries):
        try:
            r = _session.get(url, params=params, timeout=120)
            if r.status_code == 200:
                return r
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = repr(e)
        time.sleep(5 * (i + 1))
    raise SystemExit(f"GET {url} {params} failed {tries} times: {err}")


def api(path: str, **params) -> dict:
    time.sleep(0.2)
    return get(API + path, params).json()


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    return re.sub(r"[^a-z]", "", "".join(c for c in s if not unicodedata.combining(c)).lower())


def savant_names(p: dict) -> set[str]:
    """Normalised forms of a StatsAPI person's name that Savant's 'Last, First' may take."""
    forms = {p.get("lastFirstName", "")}
    for last in ("lastName", "useLastName"):
        for first in ("firstName", "useName"):
            if p.get(last) and p.get(first):
                forms.add(f"{p[last]}, {p[first]}")
    return {norm(f) for f in forms if f}


# ---------------------------------------------------------------- StatsAPI


def pick_split(splits: list, what: str) -> dict:
    """The season total: the only split, or the one without a team when a player moved."""
    if len(splits) == 1:
        return splits[0]
    total = [s for s in splits if not s.get("team")]
    if len(total) != 1:
        raise SystemExit(f"{what}: {len(splits)} splits and {len(total)} without a team")
    return total[0]


def team_ids(splits: list) -> list[int]:
    return [s["team"]["id"] for s in splits if s.get("team")]


def statsapi(seasons: list[int]) -> tuple[dict, dict]:
    """Return (bio by id, eligible {(id, season, role): stats})."""
    listed: dict[int, set[int]] = {}
    for y in seasons:
        people = api("/sports/1/players", season=y)["people"]
        if len(people) < 1000:
            raise SystemExit(f"StatsAPI season {y}: only {len(people)} players listed")
        for p in people:
            if p.get("birthCountry") == "Japan":
                listed.setdefault(p["id"], set()).add(y)
    ids = sorted(set(listed) | set(HERITAGE))
    bio = {p["id"]: p for p in api("/people", personIds=",".join(map(str, ids)))["people"]}
    if set(bio) != set(ids):
        raise SystemExit(f"StatsAPI /people did not return {sorted(set(ids) - set(bio))}")
    for pid, name in HERITAGE.items():
        if bio[pid]["fullName"] != name:
            raise SystemExit(f"heritage id {pid} is {bio[pid]['fullName']}, not {name}")
    eligible: dict[tuple[int, int, str], dict] = {}
    for y in seasons:
        teams = {t["id"]: t["abbreviation"] for t in api("/teams", sportId=1, season=y)["teams"]}
        # Every candidate is asked for every season, so a player missing from a season's roster
        # listing is still found.
        hyd = f"stats(group=[hitting,pitching],type=[season,sabermetrics],season={y},gameType=R)"
        people = api("/people", personIds=",".join(map(str, ids)), hydrate=hyd)["people"]
        if {p["id"] for p in people} != set(ids):
            raise SystemExit(f"StatsAPI /people {y} did not return {sorted(set(ids) - {p['id'] for p in people})}")
        for p in people:
            by = {}
            for st in p.get("stats", []):
                by[(st["type"]["displayName"], st["group"]["displayName"])] = st["splits"]
            for role, (group, *_rest) in ROLES.items():
                splits = by.get(("season", group))
                if not splits:
                    continue
                s = pick_split(splits, f"{p['id']} {y} {group}")
                if s.get("season") != str(y):
                    raise SystemExit(f"{p['id']} {y} {group}: StatsAPI answered season {s.get('season')}")
                stat = s["stat"]
                if role == "pitcher":
                    active = int(stat.get("numberOfPitches") or 0) > 0 or int(stat.get("battersFaced") or 0) > 0
                else:
                    active = int(stat.get("plateAppearances") or 0) > 0
                if not active:
                    continue
                saber = by.get(("sabermetrics", group))
                eligible[(p["id"], y, role)] = {
                    "stat": stat,
                    "saber": pick_split(saber, f"{p['id']} {y} saber {group}")["stat"] if saber else {},
                    "teams": [teams.get(t, str(t)) for t in team_ids(splits)],
                    "listed": y in listed.get(p["id"], set()),
                }
    for (pid, y, role), e in sorted(eligible.items()):
        if not e["listed"] and pid not in HERITAGE:
            print(f"note: {pid} {bio[pid]['fullName']} has {y} {role} games but is not on the {y} roster listing")
    return bio, eligible


# ---------------------------------------------------------------- Savant


# hyper_speed without a launch_speed, on pitches without contact: seen only in 2015-2017, at most 2
# rows per player-season-role.
STRAY_HYPER_LAST_YEAR = 2017
STRAY_HYPER_MAX = 2
CONTACT_DESCRIPTIONS = {"hit_into_play", "foul", "foul_tip", "foul_bunt", "bunt_foul_tip", "foul_pitchout"}


def fetch_savant(role: str, pid: int, season: int) -> pd.DataFrame:
    time.sleep(SAVANT_SLEEP)
    key = ROLES[role][2]
    params = {"all": "true", "type": "details", "player_type": role, "hfGT": "R|", "hfSea": f"{season}|", key: pid}
    text = get(SAVANT, params).content.decode("utf-8-sig")
    if not text.lstrip().lstrip('"').startswith("pitch_type"):
        raise SystemExit(f"Savant {role} {pid} {season}: not a Statcast CSV: {text[:200]!r}")
    # Read as text so every value is written back exactly as Savant served it.
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def check_chunk(df: pd.DataFrame, role: str, pid: int, season: int, names: set[str]) -> list[str]:
    """Checks on one player-season-role response; an empty list means it passed."""
    col = ROLES[role][1]
    errs = []
    if df.empty:
        return ["no Statcast rows"]
    for c in ("game_year", "game_type", col, "player_name", "description", "launch_speed", "hyper_speed", *KEY):
        if c not in df.columns:
            return [f"no {c} column"]
    bad = sorted(set(df["game_year"]) - {str(season)})
    if bad:
        errs.append(f"game_year {bad}")
    bad = sorted(set(df["game_type"]) - {"R"})
    if bad:
        errs.append(f"game_type {bad}")
    bad = sorted(set(df[col]) - {str(pid)})
    if bad:
        errs.append(f"{col} ids {bad[:5]}")
    got = sorted(set(df["player_name"]))
    if len(got) != 1 or norm(got[0]) not in names:
        errs.append(f"Savant player_name {got[:3]} does not match StatsAPI's name")
    for c in ALWAYS_EMPTY:
        if c in df.columns and (df[c] != "").any():
            errs.append(f"{c} is not empty ({int((df[c] != '').sum())} rows)")
    if df.duplicated(list(KEY)).any():
        errs.append(f"{int(df.duplicated(list(KEY)).sum())} duplicate {KEY} keys")
    return errs


def pitched(df: pd.DataFrame) -> int:
    """Rows that are pitches (StatsAPI's numberOfPitches counts these)."""
    return int((~df["description"].isin(NOT_PITCHES)).sum()) if len(df) else 0


def count_error(n_pitched: int, pitches: int) -> str | None:
    if abs(n_pitched - pitches) <= MAX_PITCH_DIFF:
        return None
    hint = " (Savant has not loaded every game yet?)" if n_pitched < pitches else ""
    return f"{n_pitched} Statcast pitches vs {pitches} StatsAPI pitches{hint}"


def hyper_speed_error(df: pd.DataFrame) -> str | None:
    """The column description says: hyper_speed = max(launch_speed, 88) where launch_speed is filled;
    where it is not, hyper_speed is empty or 88 (Savant fills 88 on some foul bunts, seen in 2026),
    except a few 2015-2017 pitches without contact that carry a stray value (6 rows in the
    2015-2026 build: 2015-04-25, 2016-04-24, 2016-04-28, 2016-05-08, 2016-07-10, 2017-05-02)."""
    ls = pd.to_numeric(df["launch_speed"].replace("", None), errors="coerce")
    hs = pd.to_numeric(df["hyper_speed"].replace("", None), errors="coerce")
    year = pd.to_numeric(df["game_year"], errors="coerce")
    stray = ls.isna() & hs.notna() & (hs != 88)
    bad = (ls.notna() & (hs.isna() | ((hs - ls.clip(lower=88)).abs() > 1e-9))) | (
        stray & ((year > STRAY_HYPER_LAST_YEAR) | df["description"].isin(CONTACT_DESCRIPTIONS)))
    if bad.any():
        return f"hyper_speed != max(launch_speed, 88) on {int(bad.sum())} rows"
    if stray.sum() > STRAY_HYPER_MAX:
        return f"{int(stray.sum())} rows with hyper_speed but no launch_speed (more than {STRAY_HYPER_MAX})"
    return None


def old_ids_errors(old_ids: set[int], new_ids: set[int]) -> list[str]:
    errs = []
    for i in sorted(old_ids):
        if i in CORRECTED:
            if i in new_ids:
                errs.append(f"previous id {i} is listed as wrong but is still in the build")
            if CORRECTED[i] not in new_ids:
                errs.append(f"previous id {i} was corrected to {CORRECTED[i]}, which is not in the build")
        elif i not in new_ids:
            errs.append(f"previous id {i} is not in the build and not in CORRECTED")
    return errs


# ---------------------------------------------------------------- tables


def f(x):
    return "" if x is None else x


BAT = {"pa": "plateAppearances", "ab": "atBats", "h": "hits", "hr": "homeRuns", "bb": "baseOnBalls",
       "so": "strikeOuts", "sb": "stolenBases", "avg": "avg", "obp": "obp", "slg": "slg", "ops": "ops"}
BAT_SABER = {"woba": "woba", "wrc_plus": "wRcPlus", "war": "war"}
PIT = {"gs": "gamesStarted", "ip": "inningsPitched", "bf": "battersFaced", "w": "wins", "l": "losses",
       "sv": "saves", "hld": "holds", "so": "strikeOuts", "bb": "baseOnBalls", "hr": "homeRuns",
       "era": "era", "whip": "whip"}
PIT_SABER = {"fip": "fip", "xfip": "xfip", "war": "war"}


def summary_row(bio: dict, key: tuple, e: dict, rows: int | None) -> dict:
    pid, y, role = key
    s, sab = e["stat"], e["saber"]
    row = {"mlbam_id": pid, "name": bio[pid]["fullName"], "season": y, "role": role,
           "teams": "/".join(e["teams"]), "games": f(s.get("gamesPlayed")),
           "statsapi_pitches": f(s.get("numberOfPitches")), "statcast_rows": "" if rows is None else rows}
    for k, v in BAT.items():
        row[f"bat_{k}"] = f(s.get(v)) if role == "batter" else ""
    for k, v in BAT_SABER.items():
        row[f"bat_{k}"] = f(sab.get(v)) if role == "batter" else ""
    for k, v in PIT.items():
        row[f"pit_{k}"] = f(s.get(v)) if role == "pitcher" else ""
    for k, v in PIT_SABER.items():
        row[f"pit_{k}"] = f(sab.get(v)) if role == "pitcher" else ""
    return row


def players_table(bio: dict, eligible: dict, old_ids: set[int]) -> pd.DataFrame:
    back = {v: k for k, v in CORRECTED.items()}
    first = {}
    for pid, y, _ in eligible:
        first[pid] = min(first.get(pid, 9999), y)
    rows = []
    for pid in sorted(first, key=lambda i: (first[i], bio[i]["lastName"], i)):
        p = bio[pid]
        keys = sorted(k for k in eligible if k[0] == pid)
        teams = []
        for k in keys:
            for t in eligible[k]["teams"]:
                if t not in teams:
                    teams.append(t)
        rows.append({
            "mlbam_id": pid, "name": p["fullName"], "name_last_first": p.get("lastFirstName", ""),
            "birth_country": p.get("birthCountry", ""), "birth_city": p.get("birthCity", ""),
            "birth_date": p.get("birthDate", ""),
            "heritage_note": "US-born, Japanese heritage" if pid in HERITAGE else "",
            "primary_position": p.get("primaryPosition", {}).get("abbreviation", ""),
            "bats": p.get("batSide", {}).get("code", ""), "throws": p.get("pitchHand", {}).get("code", ""),
            "mlb_debut_date": p.get("mlbDebutDate", ""),
            "roles": ", ".join(r for r in ("pitcher", "batter") if any(k[2] == r for k in keys)),
            "teams": "/".join(teams),
            "seasons": ",".join(map(str, sorted({k[1] for k in keys}))),
            "pitching_seasons": ",".join(str(k[1]) for k in keys if k[2] == "pitcher"),
            "batting_seasons": ",".join(str(k[1]) for k in keys if k[2] == "batter"),
            "in_previous_version": "yes" if (pid in VERSION1_IDS or back.get(pid) in VERSION1_IDS) else "no",
            "previous_wrong_id": back.get(pid, ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main


def parse_only(spec: str | None):
    """--only "2026,2016:547874": seasons, or season:id pairs, whose pitches are fetched (a test run)."""
    if not spec:
        return None
    sel = []
    for part in spec.split(","):
        y, _, pid = part.partition(":")
        sel.append((int(y), int(pid) if pid else None))
    return lambda pid, y: any(y == sy and (sp is None or sp == pid) for sy, sp in sel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carry", type=Path, required=True, help="folder with the current version's players.csv")
    ap.add_argument("--meta", type=Path, required=True, help="dataset-metadata.json with {placeholders}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", help="test run: only fetch pitches for these seasons / season:id pairs")
    args = ap.parse_args()
    only = parse_only(args.only)
    settings = json.loads((args.meta.parent / "settings.json").read_text(encoding="utf-8"))

    old = pd.read_csv(args.carry / PLAYERS_FILE)
    old_ids = set(old["mlbam_id"].astype(int))
    bio, eligible = statsapi(list(range(FIRST, LAST + 1)))
    new_ids = {k[0] for k in eligible}
    print(f"StatsAPI: {len(new_ids)} players, {len({k[:2] for k in eligible})} player-seasons, "
          f"{len(eligible)} player-season-roles "
          f"({sum(k[2] == 'pitcher' for k in eligible)} pitching, {sum(k[2] == 'batter' for k in eligible)} batting)")

    failures = [f"players: {e}" for e in old_ids_errors(old_ids | VERSION1_IDS, new_ids)]
    for pid, y, role in RESTORED:
        if (pid, y, role) not in eligible:
            failures.append(f"restored {pid} {y} {role} has no StatsAPI regular-season games")
    for wrong, right in CORRECTED.items():
        print(f"corrected id: {wrong} ({CORRECTED_NOTE[wrong]}) -> {right} "
              f"({bio[right]['fullName'] if right in bio else 'NOT FOUND'})")

    stage = args.out.with_name(args.out.name + ".partial")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        return build(args, only, settings, old, bio, eligible, failures, stage)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def build(args, only, settings, old, bio, eligible, failures, stage) -> int:
    old_ids = set(old["mlbam_id"].astype(int))
    columns = None
    seen_keys = {role: set() for role in ROLES}
    games: dict[tuple[int, str], dict[str, int]] = {}
    written = {role: 0 for role in ROLES}
    rows_by_key: dict[tuple, int] = {}
    todo = sorted((k for k in eligible if only is None or only(k[0], k[1])), key=lambda k: (k[1], k[2], k[0]))
    print(f"fetching {len(todo)} player-season-roles from Savant")
    for n, key in enumerate(todo, 1):
        pid, y, role = key
        df = fetch_savant(role, pid, y)
        errs = check_chunk(df, role, pid, y, savant_names(bio[pid]))
        if columns is None and not df.empty:
            columns = list(df.columns)
        if not df.empty and list(df.columns) != columns:
            errs.append("columns differ from the first Savant response")
        pitches = int(eligible[key]["stat"].get("numberOfPitches") or 0)
        if not df.empty and not errs:
            for e in (count_error(pitched(df), pitches), hyper_speed_error(df)):
                if e:
                    errs.append(e)
        if not df.empty and not errs:
            keys = df["game_pk"] + "|" + df["at_bat_number"] + "|" + df["pitch_number"]
            dup = int(keys.isin(seen_keys[role]).sum())
            if dup:
                errs.append(f"{dup} (game_pk, at_bat_number, pitch_number) keys already in the file")
            seen_keys[role].update(keys)
            mine = games.setdefault((pid, role), {})
            clash = sorted({mine[g] for g in set(df["game_pk"]) if g in mine})
            if clash:
                errs.append(f"games also in season(s) {clash}")
            mine.update(dict.fromkeys(set(df["game_pk"]), y))
        rows_by_key[key] = len(df)
        print(f"{'FAIL' if errs else 'ok  '} [{n}/{len(todo)}] {y} {role:<7} {pid} {bio[pid]['fullName']:<24} "
              f"{len(df):>5} rows / {pitches:>5} pitches" + (f"  {'; '.join(errs)}" if errs else ""), flush=True)
        failures += [f"{y} {role} {pid} {bio[pid]['fullName']}: {e}" for e in errs]
        if errs or df.empty:
            continue
        df.insert(df.columns.get_loc("player_name") + 1, "player_name_eng", bio[pid]["fullName"])
        out = stage / ROLES[role][3]
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        written[role] += len(df)
        del df

    players = players_table(bio, eligible, old_ids)
    players.to_csv(stage / PLAYERS_FILE, index=False)
    summary = pd.DataFrame([summary_row(bio, k, eligible[k], rows_by_key.get(k)) for k in sorted(eligible)])
    summary.to_csv(stage / SUMMARY_FILE, index=False)

    # Every column of every file must have a description (settings.json feeds the Kaggle metadata).
    for fname in [ROLES[r][3] for r in ROLES] + [PLAYERS_FILE, SUMMARY_FILE]:
        path = stage / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            header = next(csv.reader(fh))
        described = set(settings["files"].get(fname, {}).get("columns", {}))
        if set(header) != described:
            failures.append(f"{fname}: columns without a description {sorted(set(header) - described)}, "
                            f"described but absent {sorted(described - set(header))}")

    if failures:
        shutil.rmtree(stage, ignore_errors=True)
        print("\n".join(["", f"Gates failed ({len(failures)}), nothing written:"] + failures))
        return 1

    old_names = dict(zip(old["mlbam_id"].astype(int), old["name"]))
    new_players = players[players["in_previous_version"] == "no"]
    is_pitcher = players["primary_position"].isin(["P", "TWP"])
    pit_names = players.loc[is_pitcher, "name"].tolist()
    bat_only = players.loc[~is_pitcher, "name"].tolist()
    mop_up = players.loc[~is_pitcher & (players["pitching_seasons"] != "")]
    fill = {
        "n_players": f"{len(players)}",
        "n_player_seasons": f"{len({k[:2] for k in eligible})}",
        "n_pitching_seasons": f"{sum(k[2] == 'pitcher' for k in eligible)}",
        "n_batting_seasons": f"{sum(k[2] == 'batter' for k in eligible)}",
        "rows_pitching": f"{written['pitcher']:,}",
        "rows_batting": f"{written['batter']:,}",
        "n_pitchers": f"{len(pit_names)}",
        "n_pitching_players": f"{int((players['pitching_seasons'] != '').sum())}",
        "position_players_pitching": ", ".join(f"{r.name} ({r.pitching_seasons})" for r in mop_up.itertuples()) or "none",
        "n_position_players": f"{len(bat_only)}",
        "pitchers": ", ".join(pit_names),
        "position_players": ", ".join(bat_only),
        "rows_summary": f"{len(summary)}",
        "n_columns": f"{len(columns) + 1 if columns else 0}",
        "new_players": ", ".join(f"{r.name} ({r.mlbam_id}; {r.seasons.replace(',', ', ')})"
                                 for r in new_players.itertuples()) or "none",
        "corrections": "; ".join(f"{old_names.get(w, '?')}: {w} (that id is {CORRECTED_NOTE[w]}) -> {r}"
                                 for w, r in CORRECTED.items()),
        "heritage": " and ".join(HERITAGE.values()),
    }
    for role in ROLES:
        p = stage / ROLES[role][3]
        fill["size_" + ("pitching" if role == "pitcher" else "batting")] = f"{p.stat().st_size / 1e6:.0f} MB" if p.exists() else "0 MB"
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    for field in ("subtitle", "description"):
        text = meta[field]
        for k, v in fill.items():
            text = text.replace("{" + k + "}", v)
        left = re.findall(r"\{[a-z_]+\}", text)
        if left:
            shutil.rmtree(stage, ignore_errors=True)
            print(f"{field}: placeholders with no value {left}")
            return 1
        meta[field] = text
    if not 20 <= len(meta["subtitle"]) <= 80 or not 6 <= len(meta["title"]) <= 50:
        shutil.rmtree(stage, ignore_errors=True)
        print(f"title ({len(meta['title'])}) or subtitle ({len(meta['subtitle'])}) outside Kaggle's limits")
        return 1
    (stage / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(args.out, ignore_errors=True)
    stage.rename(args.out)
    for p in sorted(args.out.iterdir()):
        print(f"  {p.name:<30} {p.stat().st_size / 1e6:8.1f} MB")
    print(f"wrote {written['pitcher']:,} pitching rows, {written['batter']:,} batting rows, "
          f"{len(players)} players, {len(summary)} season rows to {args.out}"
          + ("  (PARTIAL test run: --only)" if only else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
