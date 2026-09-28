"""Per-player columns added to every ABS board row, joined on the MLBAM player_id.

Three free, keyless sources, each fetched for the board's own level, season and game type:

- bio: MLB StatsAPI /people (bats, throws, age in that season, height, weight, position, debut)
- api_*: MLB StatsAPI season stats (sportId 1 = MLB, 11 = Triple-A; gameType R or S):
  hitting on batter and catcher boards, pitching on pitcher boards, catching (fielding at C) on
  catcher boards
- sc_*: Baseball Savant Statcast search, aggregated by player (group_by=name), over the pitches
  of that level, season and game type: plate discipline and contact quality for batters, the
  same against pitchers, and pitches received plus zone calls for catchers
- MLB 2026 regular season only: StatsAPI sabermetrics (wOBA, wRC+, WAR) and Savant's catcher
  framing leaderboard

Every fetch is checked before it is used (enrich() returns the problems; build.py fails on any):
the season in the reply is the one asked for, regular season and spring training replies differ,
on every board Savant's search covers StatsAPI's players and agrees with StatsAPI's plate
appearances (a level or season Savant ignored would not), no zone reply is empty, and no player
appears twice in a reply.
"""
from __future__ import annotations

import datetime as dt
import io
import time

import numpy as np
import pandas as pd
import requests

API = "https://statsapi.mlb.com/api/v1"
SAVANT = "https://baseballsavant.mlb.com"
SPORT = {"mlb": 1, "aaa": 11}
SLEEP = 2.0
RETRIES = 3
PEOPLE_BATCH = 500  # 1,200 ids in one URL gives HTTP 414

# Statcast zones 11-14 are outside the strike zone, 1-9 inside.
OUT_ZONE = "&hfZ=11%7C12%7C13%7C14%7C"
IN_ZONE = "&hfZ=" + "%7C".join(str(z) for z in range(1, 10)) + "%7C"
CALLED_STRIKE = "&hfPR=called%5C.%5C.strike%7C"
BALL = "&hfPR=ball%7Cblocked%5C.%5C.ball%7C"

# Regular-season boards: share of players whose Savant plate appearances are within MAX_PA_GAP of
# StatsAPI's. The two refresh at different times, so during a season Savant can trail by a game
# (measured 2026-09-28, MLB 2026 batters: gap 0 for 55%, within 5 for 99.7%, never above 6;
# Triple-A 2025: equal for 99.55%). A level or season Savant ignored differs by hundreds.
MIN_PA_MATCH = 0.95
MAX_PA_GAP = {"batter": 8, "pitcher": 30}  # a pitcher's lag can be a whole start
# Spring training is over, so there is no lag: measured within 1 for 100% of players (exact for
# 99.8%). Swapping the 2025 and 2026 spring replies agrees within 8 for only 48%.
SPRING_PA_GAP = 1
# Share of StatsAPI's players (with a plate appearance / batter faced / inning caught) that the
# Savant reply must contain, on every board. Measured 1.0; another season covers ~62%, another
# level ~46%.
MIN_SC_COVERAGE = 0.9
# Catchers: pitches received per inning caught, median over the board. Measured 16.5-17.4 on
# every board; an empty or wrong reply would be far off (0, or a different player set).
CATCHER_PITCHES_PER_INNING = (12.0, 22.0)
# A spring reply must look like spring: the most plate appearances by anyone stays well below a
# regular season (measured: spring max 67, regular season max 726).
SPRING_MAX_PA = 150
REGULAR_MIN_MAX_PA = 400

HIT = {"gamesPlayed": "g", "plateAppearances": "pa", "atBats": "ab", "hits": "h", "doubles": "2b",
       "triples": "3b", "homeRuns": "hr", "strikeOuts": "so", "baseOnBalls": "bb", "hitByPitch": "hbp",
       "stolenBases": "sb", "avg": "avg", "obp": "obp", "slg": "slg", "ops": "ops",
       "numberOfPitches": "pitches"}
PIT = {"gamesPitched": "g", "gamesStarted": "gs", "outs": "outs", "battersFaced": "bf",
       "strikeOuts": "so", "baseOnBalls": "bb", "hitBatsmen": "hbp", "homeRuns": "hr", "era": "era",
       "whip": "whip", "strikePercentage": "strike_pct", "numberOfPitches": "pitches",
       "wildPitches": "wp"}
CATCH = {"games": "g", "innings": "innings", "stolenBases": "sb", "caughtStealing": "cs",
         "caughtStealingPercentage": "cs_pct", "passedBall": "pb", "wildPitches": "wp",
         "catcherERA": "catcher_era"}
SABER = {"woba": "woba", "wRcPlus": "wrc_plus", "war": "war"}
# Savant aggregate columns kept (renamed with an sc_ prefix).
SC_KEEP = ["pitches", "pa", "k_percent", "bb_percent", "swings", "takes", "whiffs",
           "swing_miss_percent", "woba", "xwoba", "xba", "xslg", "launch_speed", "launch_angle",
           "hardhit_percent", "barrels_per_bbe_percent", "run_exp"]


def _get(session: requests.Session, url: str, params: dict | None = None) -> requests.Response:
    for attempt in range(RETRIES):
        time.sleep(SLEEP * (1 + 2 * attempt))
        try:
            r = session.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == RETRIES - 1:
                raise
            print(f"retry {url}: {e}")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"-.--": np.nan, ".---": np.nan, "": np.nan}), errors="coerce")


def _innings(s: pd.Series) -> pd.Series:
    """Baseball notation 12.1 / 12.2 means 12 1/3 / 12 2/3 innings."""
    x = pd.to_numeric(s, errors="coerce")
    whole = np.floor(x)
    return whole + np.round((x - whole) * 10) / 3


def bio(session: requests.Session, ids: list[int]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for i in range(0, len(ids), PEOPLE_BATCH):
        chunk = ids[i:i + PEOPLE_BATCH]
        r = _get(session, f"{API}/people", {"personIds": ",".join(map(str, chunk))})
        for p in r.json().get("people", []):
            h = p.get("height") or ""
            try:
                feet, inches = h.replace('"', "").split("' ")
                height_in = int(feet) * 12 + int(inches)
            except ValueError:
                height_in = np.nan
            rows.append({
                "player_id": p["id"], "bats": (p.get("batSide") or {}).get("code"),
                "throws": (p.get("pitchHand") or {}).get("code"), "birth_date": p.get("birthDate"),
                "height_in": height_in, "weight_lb": p.get("weight"),
                "primary_position": (p.get("primaryPosition") or {}).get("abbreviation"),
                "mlb_debut_date": p.get("mlbDebutDate"), "birth_country": p.get("birthCountry"),
            })
    df = pd.DataFrame(rows).drop_duplicates("player_id")
    missing = sorted(set(ids) - set(df["player_id"]))
    errs = [f"bio: {len(missing)} player ids not returned, e.g. {missing[:5]}"] if missing else []
    return df, errs


def _season(session, group: str, level: str, year: int, game_type: str, stats: str = "season") -> list[dict]:
    r = _get(session, f"{API}/stats", dict(stats=stats, group=group, sportIds=SPORT[level], season=year,
                                           gameType=game_type, playerPool="ALL", limit=10000))
    return r.json()["stats"][0]["splits"]


def api_stats(session, level: str, year: int, game_type: str) -> tuple[dict, list[str]]:
    """{'hit': df, 'pit': df, 'catch': df} with api_* columns, one row per player."""
    out, errs = {}, []
    for key, group, fields in (("hit", "hitting", HIT), ("pit", "pitching", PIT), ("catch", "fielding", CATCH)):
        splits = _season(session, group, level, year, game_type)
        if key == "catch":
            splits = [s for s in splits if (s.get("position") or {}).get("abbreviation") == "C"]
        seasons = {s.get("season") for s in splits}
        if seasons != {str(year)}:
            errs.append(f"api {group} {level} {year} {game_type}: seasons in reply {seasons}")
        df = pd.DataFrame([{"player_id": s["player"]["id"], **{v: s["stat"].get(k) for k, v in fields.items()}}
                           for s in splits])
        if df["player_id"].duplicated().any():
            errs.append(f"api {group} {level} {year} {game_type}: {int(df['player_id'].duplicated().sum())} duplicated players")
        for c in df.columns[1:]:
            df[c] = _innings(df[c]) if c == "innings" else _num(df[c])
        if key == "hit":
            max_pa = df["pa"].max()
            if game_type == "S" and max_pa > SPRING_MAX_PA:
                errs.append(f"api hitting {level} {year} S: max PA {max_pa} > {SPRING_MAX_PA} (not spring training?)")
            if game_type == "R" and max_pa < REGULAR_MIN_MAX_PA:
                errs.append(f"api hitting {level} {year} R: max PA {max_pa} < {REGULAR_MIN_MAX_PA} (not a regular season?)")
        if key == "pit":
            df["ip"] = df["outs"] / 3
        pre = {"hit": "api_bat_", "pit": "api_pit_", "catch": "api_c_"}[key]
        out[key] = df.rename(columns={c: f"{pre}{c}" for c in df.columns if c != "player_id"})
    return out, errs


def _sc(session, level: str, year: int, game_type: str, player_type: str, extra: str = "") -> pd.DataFrame:
    minors = "minors=true&hfLevel=AAA%7C&" if level == "aaa" else ""
    url = (f"{SAVANT}/statcast_search/csv?all=true&{minors}hfGT={game_type}%7C&hfSea={year}%7C"
           f"&player_type={player_type}&group_by=name&min_pitches=0&min_results=0&min_pas=0{extra}")
    r = _get(session, url)
    text = r.content.decode("utf-8-sig")
    if not text.strip():
        if not extra:
            raise ValueError(f"Savant search returned an empty body for {level} {year} {game_type} {player_type}")
        return pd.DataFrame(columns=["player_id", "pitches", "takes", "swings", "whiffs"])
    df = pd.read_csv(io.StringIO(text))
    if df.empty and not extra:
        raise ValueError(f"Savant search returned no rows for {level} {year} {game_type} {player_type}")
    if "player_id" not in df.columns:
        raise ValueError(f"Savant search reply has no player_id column: {list(df.columns)[:8]}")
    return df


def sc_stats(session, level: str, year: int, game_type: str, api: dict) -> tuple[dict, list[str]]:
    """{'batter': df, 'pitcher': df, 'catcher': df} with sc_* columns."""
    out, errs = {}, []
    for board, ptype, api_key in (("batter", "batter", "hit"), ("pitcher", "pitcher", "pit")):
        a = _sc(session, level, year, game_type, ptype)
        oz = _sc(session, level, year, game_type, ptype, OUT_ZONE)
        iz = _sc(session, level, year, game_type, ptype, IN_ZONE)
        tag = f"sc {ptype} {level} {year} {game_type}"
        if a["player_id"].duplicated().any():
            errs.append(f"{tag}: duplicated players")
        # Level and season check: Savant's plate appearances per player against StatsAPI's.
        ref = api[api_key].set_index("player_id")["api_bat_pa" if api_key == "hit" else "api_pit_bf"]
        m = a.set_index("player_id")["pa"].to_frame().join(ref.rename("ref"), how="inner").dropna()
        gap = SPRING_PA_GAP if game_type == "S" else MAX_PA_GAP[ptype]
        share = float(((m["pa"] - m["ref"]).abs() <= gap).mean()) if len(m) else 0.0
        listed = set(ref[ref > 0].index)
        cov = len(listed & set(a["player_id"])) / len(listed) if listed else 0.0
        print(f"{tag}: plate appearances within {gap} of StatsAPI for {share:.1%} of {len(m)} players; "
              f"{cov:.1%} of StatsAPI's {len(listed)} players present")
        if share < MIN_PA_MATCH:
            errs.append(f"{tag}: plate appearances within {gap} of StatsAPI for {share:.1%} of {len(m)} players < {MIN_PA_MATCH:.0%}")
        if cov < MIN_SC_COVERAGE:
            errs.append(f"{tag}: only {cov:.1%} of StatsAPI's players present < {MIN_SC_COVERAGE:.0%}")
        keep = [c for c in SC_KEEP if c in a.columns]
        df = a[["player_id", *keep]].copy()
        for z, zdf in (("oz", oz), ("iz", iz)):
            zi = zdf.set_index("player_id")
            for c in ("pitches", "swings", "whiffs"):
                df[f"{z}_{c}"] = df["player_id"].map(zi[c]).fillna(0)
        if (df["oz_pitches"] + df["iz_pitches"] > df["pitches"]).any():
            errs.append(f"{tag}: zone pitches exceed pitches")
        for name in ("oz_pitches", "iz_pitches", "oz_swings", "iz_swings"):
            if df[name].sum() == 0:
                errs.append(f"{tag}: {name} is 0 for every player (empty reply)")
        if ((df["oz_swings"] > df["oz_pitches"]) | (df["iz_swings"] > df["iz_pitches"])).any():
            errs.append(f"{tag}: more swings than pitches in a zone")
        # Pitches with a Statcast zone (tracking missing for some pitches, notably MLB 2025 spring).
        df["zone_tracked_percent"] = 100 * (df["oz_pitches"] + df["iz_pitches"]) / df["pitches"].where(df["pitches"] > 0)
        df["chase_percent"] = 100 * df["oz_swings"] / df["oz_pitches"].where(df["oz_pitches"] > 0)
        df["zone_swing_percent"] = 100 * df["iz_swings"] / df["iz_pitches"].where(df["iz_pitches"] > 0)
        df["zone_contact_percent"] = 100 * (1 - df["iz_whiffs"] / df["iz_swings"].where(df["iz_swings"] > 0))
        df["chase_contact_percent"] = 100 * (1 - df["oz_whiffs"] / df["oz_swings"].where(df["oz_swings"] > 0))
        pre = "sc_bat_" if board == "batter" else "sc_pit_"
        out[board] = df.rename(columns={c: f"{pre}{c}" for c in df.columns if c != "player_id"})
    # Catchers: pitches received and how the umpire called takes by zone (as recorded by Statcast).
    tag = f"sc fielder_2 {level} {year} {game_type}"
    base = _sc(session, level, year, game_type, "fielder_2")
    if base["player_id"].duplicated().any():
        errs.append(f"{tag}: duplicated players")
    parts = {"oz_takes": ("takes", OUT_ZONE), "oz_called_strikes": ("pitches", OUT_ZONE + CALLED_STRIKE),
             "iz_takes": ("takes", IN_ZONE), "iz_called_balls": ("pitches", IN_ZONE + BALL)}
    df = base[["player_id", "pitches", "takes"]].copy()
    for name, (col, extra) in parts.items():
        d = _sc(session, level, year, game_type, "fielder_2", extra)
        df = df.join(d.set_index("player_id")[col].rename(name), on="player_id")
        df[name] = df[name].fillna(0)
    if ((df["oz_called_strikes"] > df["oz_takes"]) | (df["iz_called_balls"] > df["iz_takes"])
            | (df["oz_takes"] + df["iz_takes"] > df["takes"])).any():
        errs.append(f"{tag}: zone counts inconsistent (called > takes, or zone takes > takes)")
    for name in parts:
        if df[name].sum() == 0:
            errs.append(f"{tag}: {name} is 0 for every catcher (empty reply)")
    inn = api["catch"].set_index("player_id")["api_c_innings"]
    listed = set(inn[inn > 0].index)
    cov = len(listed & set(df["player_id"])) / len(listed) if listed else 0.0
    ppi = (df.set_index("player_id")["pitches"] / inn[inn >= 9]).dropna().median()
    lo, hi = CATCHER_PITCHES_PER_INNING
    print(f"{tag}: {cov:.1%} of StatsAPI's {len(listed)} catchers present; median pitches received per inning {ppi:.1f}")
    if cov < MIN_SC_COVERAGE:
        errs.append(f"{tag}: only {cov:.1%} of StatsAPI's catchers present < {MIN_SC_COVERAGE:.0%}")
    if not lo <= ppi <= hi:
        errs.append(f"{tag}: median pitches received per inning {ppi:.1f} outside {lo}-{hi}")
    df["zone_tracked_percent"] = 100 * (df["oz_takes"] + df["iz_takes"]) / df["takes"].where(df["takes"] > 0)
    df["oz_called_strike_percent"] = 100 * df["oz_called_strikes"] / df["oz_takes"].where(df["oz_takes"] > 0)
    df["iz_called_ball_percent"] = 100 * df["iz_called_balls"] / df["iz_takes"].where(df["iz_takes"] > 0)
    df = df.rename(columns={"pitches": "pitches_received", "takes": "takes_received"})
    out["catcher"] = df.rename(columns={c: f"sc_c_{c}" for c in df.columns if c != "player_id"})
    return out, errs


def mlb2026_extras(session) -> tuple[dict, list[str]]:
    """Sabermetrics (MLB 2026 regular season, hitting and pitching) and catcher framing."""
    errs, out = [], {}
    for key, group in (("hit", "hitting"), ("pit", "pitching")):
        splits = _season(session, group, "mlb", 2026, "R", stats="sabermetrics")
        seasons = {x.get("season") for x in splits}
        if seasons != {"2026"}:
            errs.append(f"sabermetrics {group}: seasons in reply {seasons}")
        fields = SABER if key == "hit" else {"war": "war", "fip": "fip", "xfip": "xfip"}
        df = pd.DataFrame([{"player_id": s["player"]["id"], **{v: s["stat"].get(k) for k, v in fields.items()}}
                           for s in splits])
        if df.empty or df["player_id"].duplicated().any():
            errs.append(f"sabermetrics {group}: {len(df)} rows or duplicated players")
        for c in df.columns[1:]:
            df[c] = _num(df[c])
        pre = "api_bat_" if key == "hit" else "api_pit_"
        out[key] = df.rename(columns={c: f"{pre}{c}" for c in df.columns if c != "player_id"})
    frames = {}
    for y in (2025, 2026):
        r = _get(session, f"{SAVANT}/leaderboard/catcher-framing",
                 {"type": "catcher", "seasonStart": y, "seasonEnd": y, "team": "", "min": "q", "csv": "true"})
        frames[y] = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
    f = frames[2026]
    id_col = next((c for c in ("id", "player_id") if c in f.columns), None)
    if id_col is None or "rv_tot" not in f.columns:
        errs.append(f"catcher framing: unexpected columns {list(f.columns)[:10]}")
        out["framing"] = pd.DataFrame(columns=["player_id"])
    else:
        # The reply has no season column: 2025 and 2026 must differ, or the season was ignored.
        if frames[2025].equals(f):
            errs.append("catcher framing: the 2025 and 2026 replies are identical (season ignored)")
        keep = {id_col: "player_id", "rv_tot": "framing_runs", "pct_tot": "framing_strike_rate",
                "n_called_pitches": "framing_called_pitches"}
        out["framing"] = f[[c for c in keep if c in f.columns]].rename(columns=keep).rename(
            columns=lambda c: c if c == "player_id" else f"sc_c_{c}")
    return out, errs


def age_in_season(birth: pd.Series, year: pd.Series) -> pd.Series:
    """Age on June 30 of the season (MLB's convention for a season's age)."""
    b = pd.to_datetime(birth, errors="coerce")
    ref = pd.to_datetime(year.astype(str) + "-06-30")
    return (ref.dt.year - b.dt.year - ((ref.dt.month < b.dt.month) | ((ref.dt.month == b.dt.month) & (ref.dt.day < b.dt.day)))).astype("Int64")


def enrich(boards: dict, session: requests.Session | None = None) -> tuple[dict, list[str]]:
    """Return the boards with the added columns (same row order) and a list of problems."""
    session = session or requests.Session()
    errs = []
    ids = sorted({int(i) for df in boards.values() for i in df["player_id"]})
    people, e = bio(session, ids)
    errs += e
    extras, e = mlb2026_extras(session)
    errs += e
    out = {}
    for level, year, game_type in sorted({k[:3] for k in boards}):
        api, e = api_stats(session, level, year, game_type)
        errs += e
        sc, e = sc_stats(session, level, year, game_type, api)
        errs += e
        for challenger in ("batter", "pitcher", "catcher"):
            key = (level, year, game_type, challenger)
            if key not in boards:
                continue
            df = boards[key].merge(people, on="player_id", how="left", validate="many_to_one")
            df.insert(df.columns.get_loc("birth_date") + 1, "age", age_in_season(df["birth_date"], df["year"]))
            parts = [api["pit"] if challenger == "pitcher" else api["hit"], sc[challenger]]
            if challenger == "catcher":
                parts.append(api["catch"])
            if (level, year, game_type) == ("mlb", 2026, "R"):
                parts.append(extras["pit"] if challenger == "pitcher" else extras["hit"])
                if challenger == "catcher":
                    parts.append(extras["framing"])
            for p in parts:
                df = df.merge(p, on="player_id", how="left", validate="many_to_one")
            if len(df) != len(boards[key]):
                errs.append(f"{key}: rows changed while adding columns")
            out[key] = df
    return out, errs
