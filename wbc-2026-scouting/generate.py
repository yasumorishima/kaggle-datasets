"""WBC 2026 Scouting Dataset - Generation Script

This script reads WBC 2026 roster data and per-country Statcast CSVs,
and produces the clean dataset files published here.

Data source:
  Baseball Savant (https://baseballsavant.mlb.com/) via pybaseball
  WBC 2026 official roster (Baseball America, February 2026)

Usage:
  python generate.py --wbc-dir /path/to/wbc-scouting --out-dir /path/to/output

Requirements:
  pip install pandas numpy pybaseball

Input directory layout expected (wbc-scouting/data/):
  <country>_statcast.csv          -- batter pitch-level data
  <country>_pitchers_statcast.csv -- pitcher pitch-level data
  wbc2026_rosters.csv             -- full roster (Japanese semi-structured CSV)
  mlbstadiums_wbc.csv             -- stadium coordinates
"""

import argparse
import pathlib
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Country mapping: CSV prefix -> (ISO code, country name, pool)
# ---------------------------------------------------------------------------
COUNTRY_MAP = {
    "usa":       ("USA", "USA",                  "B (Houston)"),
    "japan":     ("JPN", "Japan",                "C (Tokyo)"),
    "dr":        ("DOM", "Dominican Republic",   "D (Miami)"),
    "venezuela": ("VEN", "Venezuela",            "D (Miami)"),
    "pr":        ("PUR", "Puerto Rico",          "A (San Juan)"),
    "mex":       ("MEX", "Mexico",               "B (Houston)"),
    "kor":       ("KOR", "Korea",                "C (Tokyo)"),
    "twn":       ("TPE", "Chinese Taipei",       "C (Tokyo)"),
    "ned":       ("NED", "Netherlands",          "D (Miami)"),
    "cuba":      ("CUB", "Cuba",                 "A (San Juan)"),
    "can":       ("CAN", "Canada",               "A (San Juan)"),
    "ita":       ("ITA", "Italy",                "B (Houston)"),
    "isr":       ("ISR", "Israel",               "D (Miami)"),
    "gb":        ("GBR", "Great Britain",        "B (Houston)"),
    "pan":       ("PAN", "Panama",               "A (San Juan)"),
    "col":       ("COL", "Colombia",             "A (San Juan)"),
    "nic":       ("NCA", "Nicaragua",            "D (Miami)"),
    "aus":       ("AUS", "Australia",            "C (Tokyo)"),
    "bra":       ("BRA", "Brazil",               "B (Houston)"),
    "cze":       ("CZE", "Czechia",              "C (Tokyo)"),
}

# Roster CSV country header -> ISO code
ROSTER_COUNTRY_MAP = {
    "USA": "USA", "Japan": "JPN", "Dominican Rep.": "DOM",
    "Venezuela": "VEN", "Puerto Rico": "PUR", "Mexico": "MEX",
    "Korea": "KOR", "Chinese Taipei": "TPE", "Netherlands": "NED",
    "Cuba": "CUB", "Canada": "CAN", "Italy": "ITA",
    "Israel": "ISR", "Great Britain": "GBR", "Panama": "PAN",
    "Colombia": "COL", "Nicaragua": "NCA", "Australia": "AUS",
    "Brazil": "BRA", "Czechia": "CZE",
}

# Pool city (Japanese) -> English
POOL_CITY_MAP = {
    "\u30b5\u30f3\u30d5\u30a2\u30f3": "San Juan",
    "\u30d2\u30e5\u30fc\u30b9\u30c8\u30f3": "Houston",
    "\u6771\u4eac\u30c9\u30fc\u30e0": "Tokyo",
    "\u30de\u30a4\u30a2\u30df": "Miami",
}

# MLB team name (Japanese) -> English
MLB_TEAM_MAP = {
    "\u30a2\u30b9\u30ec\u30c1\u30c3\u30af\u30b9": "Athletics",
    "\u30a2\u30c8\u30e9\u30f3\u30bf\u30fb\u30d6\u30ec\u30fc\u30d6\u30b9": "Atlanta Braves",
    "\u30a2\u30ea\u30be\u30ca\u30fb\u30c0\u30a4\u30e4\u30e2\u30f3\u30c9\u30d0\u30c3\u30af\u30b9": "Arizona Diamondbacks",
    "\u30ab\u30f3\u30b6\u30b9\u30b7\u30c6\u30a3\u30fb\u30ed\u30a4\u30e4\u30eb\u30ba": "Kansas City Royals",
    "\u30af\u30ea\u30fc\u30d6\u30e9\u30f3\u30c9\u30fb\u30ac\u30fc\u30c7\u30a3\u30a2\u30f3\u30ba": "Cleveland Guardians",
    "\u30b3\u30ed\u30e9\u30c9\u30fb\u30ed\u30c3\u30ad\u30fc\u30ba": "Colorado Rockies",
    "\u30b5\u30f3\u30c7\u30a3\u30a8\u30b4\u30fb\u30d1\u30c9\u30ec\u30b9": "San Diego Padres",
    "\u30b5\u30f3\u30d5\u30e9\u30f3\u30b7\u30b9\u30b3\u30fb\u30b8\u30e3\u30a4\u30a2\u30f3\u30c4": "San Francisco Giants",
    "\u30b7\u30a2\u30c8\u30eb\u30fb\u30de\u30ea\u30ca\u30fc\u30ba": "Seattle Mariners",
    "\u30b7\u30ab\u30b4\u30fb\u30ab\u30d6\u30b9": "Chicago Cubs",
    "\u30b7\u30ab\u30b4\u30fb\u30db\u30ef\u30a4\u30c8\u30bd\u30c3\u30af\u30b9": "Chicago White Sox",
    "\u30b7\u30f3\u30b7\u30ca\u30c6\u30a3\u30fb\u30ec\u30c3\u30ba": "Cincinnati Reds",
    "\u30bb\u30f3\u30c8\u30eb\u30a4\u30b9\u30fb\u30ab\u30fc\u30b8\u30ca\u30eb\u30b9": "St. Louis Cardinals",
    "\u30bf\u30f3\u30d1\u30d9\u30a4\u30fb\u30ec\u30a4\u30ba": "Tampa Bay Rays",
    "\u30c6\u30ad\u30b5\u30b9\u30fb\u30ec\u30f3\u30b8\u30e3\u30fc\u30ba": "Texas Rangers",
    "\u30c7\u30c8\u30ed\u30a4\u30c8\u30fb\u30bf\u30a4\u30ac\u30fc\u30b9": "Detroit Tigers",
    "\u30c8\u30ed\u30f3\u30c8\u30fb\u30d6\u30eb\u30fc\u30b8\u30a7\u30a4\u30ba": "Toronto Blue Jays",
    "\u30cb\u30e5\u30fc\u30e8\u30fc\u30af\u30fb\u30e1\u30c3\u30c4": "New York Mets",
    "\u30cb\u30e5\u30fc\u30e8\u30fc\u30af\u30fb\u30e4\u30f3\u30ad\u30fc\u30b9": "New York Yankees",
    "\u30d2\u30e5\u30fc\u30b9\u30c8\u30f3\u30fb\u30a2\u30b9\u30c8\u30ed\u30ba": "Houston Astros",
    "\u30d4\u30c3\u30c4\u30d0\u30fc\u30b0\u30fb\u30d1\u30a4\u30ec\u30fc\u30c4": "Pittsburgh Pirates",
    "\u30d5\u30a3\u30e9\u30c7\u30eb\u30d5\u30a3\u30a2\u30fb\u30d5\u30a3\u30ea\u30fc\u30ba": "Philadelphia Phillies",
    "\u30dc\u30b9\u30c8\u30f3\u30fb\u30ec\u30c3\u30c9\u30bd\u30c3\u30af\u30b9": "Boston Red Sox",
    "\u30dc\u30eb\u30c6\u30a3\u30e2\u30a2\u30fb\u30aa\u30ea\u30aa\u30fc\u30eb\u30ba": "Baltimore Orioles",
    "\u30de\u30a4\u30a2\u30df\u30fb\u30de\u30fc\u30ea\u30f3\u30ba": "Miami Marlins",
    "\u30df\u30cd\u30bd\u30bf\u30fb\u30c4\u30a4\u30f3\u30ba": "Minnesota Twins",
    "\u30df\u30eb\u30a6\u30a9\u30fc\u30ad\u30fc\u30fb\u30d6\u30eb\u30ef\u30fc\u30ba": "Milwaukee Brewers",
    "\u30ed\u30b5\u30f3\u30bc\u30eb\u30b9\u30fb\u30a8\u30f3\u30bc\u30eb\u30b9": "Los Angeles Angels",
    "\u30ed\u30b5\u30f3\u30bc\u30eb\u30b9\u30fb\u30c9\u30b8\u30e3\u30fc\u30b9": "Los Angeles Dodgers",
    "\u30ef\u30b7\u30f3\u30c8\u30f3\u30fb\u30ca\u30b7\u30e7\u30ca\u30eb\u30ba": "Washington Nationals",
    "\u30aa\u30ea\u30aa\u30fc\u30eb\u30ba": "Orioles",
    "\u30e4\u30f3\u30ad\u30fc\u30b9": "Yankees",
    "\u30c9\u30b8\u30e3\u30fc\u30b9": "Dodgers",
}


def translate_team(team_ja: str) -> str:
    """Translate a Japanese MLB team name to English."""
    if team_ja in MLB_TEAM_MAP:
        return MLB_TEAM_MAP[team_ja]
    if "FA" in team_ja:
        m = re.search(r"\u524d(.+?)[\uff09)]", team_ja)
        if m:
            prev = MLB_TEAM_MAP.get(m.group(1), m.group(1))
            return f"FA (former {prev})"
        return "Free Agent"
    if "\u5f15\u9000\u5fa9\u5e30" in team_ja:
        m = re.search(r"\u524d(.+?)[\uff09)]", team_ja)
        if m:
            prev = MLB_TEAM_MAP.get(m.group(1), m.group(1))
            return f"Returning from retirement (former {prev})"
        return "Returning from retirement"
    return team_ja


# ---------------------------------------------------------------------------
# 1. Parse roster CSV
# ---------------------------------------------------------------------------
def parse_roster(data_dir: pathlib.Path) -> pd.DataFrame:
    """Parse the semi-structured wbc2026_rosters.csv into a clean table."""
    raw = (data_dir / "wbc2026_rosters.csv").read_text(encoding="utf-8")
    lines = raw.strip().split("\n")

    rows = []
    current_country = None
    current_pool = None
    country_pattern = re.compile(
        r"^(.+?)\uff08.+?\uff09\s*\u2014\s*\u30d7\u30fc\u30eb([A-D])\uff08(.+?)\uff09\s*\u2014"
    )

    for line in lines:
        line = line.strip()
        if not line or line.startswith("WBC 2026") or line.startswith("\u51fa\u5178"):
            continue
        m = country_pattern.match(line)
        if m:
            country_name = m.group(1).strip()
            pool_letter = m.group(2)
            pool_city = m.group(3)
            current_country = ROSTER_COUNTRY_MAP.get(country_name)
            city_en = POOL_CITY_MAP.get(pool_city, pool_city)
            current_pool = f"{pool_letter} ({city_en})"
            continue
        if line.startswith("\u9078\u624b\u540d,"):
            continue
        parts = line.split(",")
        if len(parts) >= 3 and current_country and parts[0].strip():
            name = parts[0].strip()
            pos = parts[1].strip() if len(parts) > 1 else ""
            team = translate_team(parts[2].strip()) if len(parts) > 2 else ""
            on_40man = parts[3].strip() if len(parts) > 3 else ""
            if " / " in name:
                _, name_en = name.split(" / ", 1)
            elif "\uff08" in name:
                name_en = re.sub(r"\uff08.+?\uff09", "", name).strip()
            else:
                name_en = name
            role = "pitcher" if pos in ("RHP", "LHP") else "batter"
            rows.append({
                "name": name_en,
                "country": current_country,
                "pool": current_pool,
                "position": pos,
                "team": team,
                "on_40_man": on_40man == "YES",
                "role": role,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Combine raw Statcast CSVs
# ---------------------------------------------------------------------------
def combine_statcast(data_dir: pathlib.Path, pattern_suffix: str,
                     exclude_pattern: str = "") -> pd.DataFrame:
    """Combine all Statcast CSVs matching pattern into one DataFrame."""
    frames = []
    for csv_path in sorted(data_dir.glob(f"*{pattern_suffix}")):
        if exclude_pattern and exclude_pattern in csv_path.name:
            continue
        stem = csv_path.stem
        prefix = stem.replace(pattern_suffix.replace(".csv", ""), "").rstrip("_")
        if prefix not in COUNTRY_MAP:
            print(f"  SKIP unknown prefix: {prefix} ({csv_path.name})")
            continue
        iso, country_name, _ = COUNTRY_MAP[prefix]
        df = pd.read_csv(csv_path)
        df["country"] = iso
        df["country_name"] = country_name
        frames.append(df)
        print(f"  {csv_path.name}: {len(df):,} rows ({iso})")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. Compute summary stats
# ---------------------------------------------------------------------------
def batting_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player batting summary from Statcast event-level data."""
    AB_EVENTS = {
        "single", "double", "triple", "home_run", "field_out",
        "strikeout", "grounded_into_double_play", "double_play",
        "force_out", "fielders_choice", "fielders_choice_out",
        "strikeout_double_play", "triple_play", "field_error",
    }
    PA_EVENTS = AB_EVENTS | {
        "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
        "sac_fly_double_play", "catcher_interf", "intent_walk",
    }
    HIT_EVENTS = {"single", "double", "triple", "home_run"}

    rows = []
    for (batter_id, country), gdf in df.groupby(["batter", "country"]):
        name = gdf["player_name"].iloc[0]
        evts = gdf.dropna(subset=["events"])
        pa = evts[evts["events"].isin(PA_EVENTS)]
        ab = evts[evts["events"].isin(AB_EVENTS)]
        hits = evts[evts["events"].isin(HIT_EVENTS)]

        n_pa   = len(pa)
        n_ab   = len(ab)
        n_h    = len(hits)
        n_bb   = len(pa[pa["events"].isin({"walk", "intent_walk"})])
        n_hbp  = len(pa[pa["events"] == "hit_by_pitch"])
        n_1b   = len(hits[hits["events"] == "single"])
        n_2b   = len(hits[hits["events"] == "double"])
        n_3b   = len(hits[hits["events"] == "triple"])
        n_hr   = len(hits[hits["events"] == "home_run"])
        n_k    = len(ab[ab["events"].str.contains("strikeout", na=False)])
        tb     = n_1b + 2 * n_2b + 3 * n_3b + 4 * n_hr
        avg    = n_h / n_ab if n_ab else np.nan
        obp    = (n_h + n_bb + n_hbp) / n_pa if n_pa else np.nan
        slg    = tb / n_ab if n_ab else np.nan
        ops    = obp + slg if not (pd.isna(obp) or pd.isna(slg)) else np.nan
        k_pct  = n_k / n_pa * 100 if n_pa else np.nan
        bb_pct = n_bb / n_pa * 100 if n_pa else np.nan
        xwoba  = gdf["estimated_woba_using_speedangle"].dropna().mean()
        avg_ev = gdf["launch_speed"].dropna().mean()
        avg_la = gdf["launch_angle"].dropna().mean()

        rows.append({
            "mlbam_id": int(batter_id), "player_name": name, "country": country,
            "PA": n_pa, "AB": n_ab, "H": n_h,
            "1B": n_1b, "2B": n_2b, "3B": n_3b, "HR": n_hr,
            "BB": n_bb, "HBP": n_hbp, "K": n_k, "TB": tb,
            "AVG":   round(avg,   3) if not pd.isna(avg)   else np.nan,
            "OBP":   round(obp,   3) if not pd.isna(obp)   else np.nan,
            "SLG":   round(slg,   3) if not pd.isna(slg)   else np.nan,
            "OPS":   round(ops,   3) if not pd.isna(ops)   else np.nan,
            "K_pct": round(k_pct,  1) if not pd.isna(k_pct)  else np.nan,
            "BB_pct": round(bb_pct, 1) if not pd.isna(bb_pct) else np.nan,
            "xwOBA": round(xwoba,  3) if not pd.isna(xwoba)  else np.nan,
            "avg_exit_velo":    round(avg_ev, 1) if not pd.isna(avg_ev) else np.nan,
            "avg_launch_angle": round(avg_la, 1) if not pd.isna(avg_la) else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["country", "player_name"]).reset_index(drop=True)


def pitching_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player pitching summary from Statcast pitch-level data."""
    AB_EVENTS = {
        "single", "double", "triple", "home_run", "field_out",
        "strikeout", "grounded_into_double_play", "double_play",
        "force_out", "fielders_choice", "fielders_choice_out",
        "strikeout_double_play", "triple_play", "field_error",
    }
    PA_EVENTS = AB_EVENTS | {
        "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
        "sac_fly_double_play", "catcher_interf", "intent_walk",
    }
    HIT_EVENTS = {"single", "double", "triple", "home_run"}

    rows = []
    for (pitcher_id, country), gdf in df.groupby(["pitcher", "country"]):
        name = gdf["player_name"].iloc[0]
        total_pitches = len(gdf)
        evts  = gdf.dropna(subset=["events"])
        pa    = evts[evts["events"].isin(PA_EVENTS)]
        ab    = evts[evts["events"].isin(AB_EVENTS)]
        hits  = evts[evts["events"].isin(HIT_EVENTS)]

        n_pa  = len(pa)
        n_ab  = len(ab)
        n_h   = len(hits)
        n_k   = len(ab[ab["events"].str.contains("strikeout", na=False)])
        n_bb  = len(pa[pa["events"].isin({"walk", "intent_walk"})])
        n_hr  = len(hits[hits["events"] == "home_run"])
        n_1b  = len(hits[hits["events"] == "single"])
        n_2b  = len(hits[hits["events"] == "double"])
        n_3b  = len(hits[hits["events"] == "triple"])
        tb    = n_1b + 2 * n_2b + 3 * n_3b + 4 * n_hr
        opp_avg = n_h / n_ab if n_ab else np.nan
        opp_slg = tb / n_ab if n_ab else np.nan
        k_pct   = n_k / n_pa * 100 if n_pa else np.nan
        bb_pct  = n_bb / n_pa * 100 if n_pa else np.nan
        xwoba   = gdf["estimated_woba_using_speedangle"].dropna().mean()
        avg_velo = gdf["release_speed"].dropna().mean() if "release_speed" in gdf.columns else np.nan
        avg_spin = gdf["release_spin_rate"].dropna().mean() if "release_spin_rate" in gdf.columns else np.nan
        pitch_counts  = gdf["pitch_type"].value_counts()
        top_pitch     = pitch_counts.index[0] if len(pitch_counts) > 0 else ""
        n_pitch_types = len(pitch_counts)

        rows.append({
            "mlbam_id": int(pitcher_id), "player_name": name, "country": country,
            "total_pitches": total_pitches, "PA_faced": n_pa,
            "K": n_k, "BB": n_bb, "HR_allowed": n_hr, "H_allowed": n_h,
            "opp_AVG": round(opp_avg, 3) if not pd.isna(opp_avg) else np.nan,
            "opp_SLG": round(opp_slg, 3) if not pd.isna(opp_slg) else np.nan,
            "K_pct":   round(k_pct,  1) if not pd.isna(k_pct)  else np.nan,
            "BB_pct":  round(bb_pct, 1) if not pd.isna(bb_pct) else np.nan,
            "xwOBA_against":  round(xwoba,    3) if not pd.isna(xwoba)    else np.nan,
            "avg_velo":       round(avg_velo,  1) if not pd.isna(avg_velo) else np.nan,
            "avg_spin_rate":  round(avg_spin,  0) if not pd.isna(avg_spin) else np.nan,
            "pitch_type_count": n_pitch_types,
            "primary_pitch": top_pitch,
        })

    return pd.DataFrame(rows).sort_values(["country", "player_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate WBC 2026 Kaggle dataset CSVs")
    parser.add_argument("--wbc-dir", required=True,
                        help="Path to wbc-scouting repo root (must contain data/ subdirectory)")
    parser.add_argument("--out-dir", default=".",
                        help="Output directory for generated CSVs (default: current directory)")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.wbc_dir) / "data"
    out_dir  = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== WBC 2026 Kaggle Dataset Generator ===\n")

    print("[1/6] Parsing roster...")
    roster = parse_roster(data_dir)
    roster.to_csv(out_dir / "rosters.csv", index=False)
    print(f"  -> rosters.csv: {len(roster)} players, {roster['country'].nunique()} countries\n")

    print("[2/6] Combining batter Statcast CSVs...")
    batter_raw = combine_statcast(data_dir, "_statcast.csv", exclude_pattern="_pitchers_")
    if not batter_raw.empty:
        batter_raw.to_csv(out_dir / "statcast_batters.csv", index=False)
        print(f"  -> statcast_batters.csv: {len(batter_raw):,} rows\n")

    print("[3/6] Combining pitcher Statcast CSVs...")
    pitcher_raw = combine_statcast(data_dir, "_pitchers_statcast.csv")
    if not pitcher_raw.empty:
        pitcher_raw.to_csv(out_dir / "statcast_pitchers.csv", index=False)
        print(f"  -> statcast_pitchers.csv: {len(pitcher_raw):,} rows\n")

    print("[4/6] Computing batter summary stats...")
    if not batter_raw.empty:
        bat_summary = batting_summary(batter_raw)
        bat_summary.to_csv(out_dir / "batter_summary.csv", index=False)
        print(f"  -> batter_summary.csv: {len(bat_summary)} players\n")

    print("[5/6] Computing pitcher summary stats...")
    if not pitcher_raw.empty:
        pit_summary = pitching_summary(pitcher_raw)
        pit_summary.to_csv(out_dir / "pitcher_summary.csv", index=False)
        print(f"  -> pitcher_summary.csv: {len(pit_summary)} players\n")

    print("[6/6] Copying stadium data...")
    stadiums_src = data_dir / "mlbstadiums_wbc.csv"
    if stadiums_src.exists():
        stadiums = pd.read_csv(stadiums_src)
        if "Unnamed: 0" in stadiums.columns:
            stadiums = stadiums.drop(columns=["Unnamed: 0"])
        stadiums.to_csv(out_dir / "stadiums.csv", index=False)
        print(f"  -> stadiums.csv: {len(stadiums):,} rows\n")

    print("=== Done! ===")


if __name__ == "__main__":
    main()
