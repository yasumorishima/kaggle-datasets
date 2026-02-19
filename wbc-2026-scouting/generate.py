"""Generate WBC 2026 Kaggle dataset CSVs.

Reads roster data and Statcast CSVs from the wbc-scouting repo,
produces clean dataset files for Kaggle upload.

Usage:
    python generate.py
"""

import pathlib
import re

import numpy as np
import pandas as pd

WBC_DIR = pathlib.Path(r"C:\Users\fw_ya\Desktop\Claude_code\wbc-scouting")
DATA_DIR = WBC_DIR / "data"
OUT_DIR = pathlib.Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Country mapping: CSV prefix -> (country_code, country_name, pool)
# ---------------------------------------------------------------------------
COUNTRY_MAP = {
    "usa": ("USA", "USA", "B (Houston)"),
    "japan": ("JPN", "Japan", "C (Tokyo)"),
    "dr": ("DOM", "Dominican Republic", "D (Miami)"),
    "venezuela": ("VEN", "Venezuela", "D (Miami)"),
    "pr": ("PUR", "Puerto Rico", "A (San Juan)"),
    "mex": ("MEX", "Mexico", "B (Houston)"),
    "kor": ("KOR", "Korea", "C (Tokyo)"),
    "twn": ("TPE", "Chinese Taipei", "C (Tokyo)"),
    "ned": ("NED", "Netherlands", "D (Miami)"),
    "cuba": ("CUB", "Cuba", "A (San Juan)"),
    "can": ("CAN", "Canada", "A (San Juan)"),
    "ita": ("ITA", "Italy", "B (Houston)"),
    "isr": ("ISR", "Israel", "D (Miami)"),
    "gb": ("GBR", "Great Britain", "B (Houston)"),
    "pan": ("PAN", "Panama", "A (San Juan)"),
    "col": ("COL", "Colombia", "A (San Juan)"),
    "nic": ("NCA", "Nicaragua", "D (Miami)"),
    "aus": ("AUS", "Australia", "C (Tokyo)"),
    "bra": ("BRA", "Brazil", "B (Houston)"),
    "cze": ("CZE", "Czechia", "C (Tokyo)"),
}

# Pool city Japanese -> English
POOL_CITY_MAP = {
    "サンフアン": "San Juan",
    "ヒューストン": "Houston",
    "東京ドーム": "Tokyo",
    "マイアミ": "Miami",
}

# MLB team name Japanese -> English
MLB_TEAM_MAP = {
    "アスレチックス": "Athletics",
    "アトランタ・ブレーブス": "Atlanta Braves",
    "アリゾナ・ダイヤモンドバックス": "Arizona Diamondbacks",
    "カンザスシティ・ロイヤルズ": "Kansas City Royals",
    "クリーブランド・ガーディアンズ": "Cleveland Guardians",
    "コロラド・ロッキーズ": "Colorado Rockies",
    "サンディエゴ・パドレス": "San Diego Padres",
    "サンフランシスコ・ジャイアンツ": "San Francisco Giants",
    "シアトル・マリナーズ": "Seattle Mariners",
    "シカゴ・カブス": "Chicago Cubs",
    "シカゴ・ホワイトソックス": "Chicago White Sox",
    "シンシナティ・レッズ": "Cincinnati Reds",
    "セントルイス・カージナルス": "St. Louis Cardinals",
    "タンパベイ・レイズ": "Tampa Bay Rays",
    "テキサス・レンジャーズ": "Texas Rangers",
    "デトロイト・タイガース": "Detroit Tigers",
    "トロント・ブルージェイズ": "Toronto Blue Jays",
    "ニューヨーク・メッツ": "New York Mets",
    "ニューヨーク・ヤンキース": "New York Yankees",
    "ヒューストン・アストロズ": "Houston Astros",
    "ピッツバーグ・パイレーツ": "Pittsburgh Pirates",
    "フィラデルフィア・フィリーズ": "Philadelphia Phillies",
    "ボストン・レッドソックス": "Boston Red Sox",
    "ボルティモア・オリオールズ": "Baltimore Orioles",
    "マイアミ・マーリンズ": "Miami Marlins",
    "ミネソタ・ツインズ": "Minnesota Twins",
    "ミルウォーキー・ブルワーズ": "Milwaukee Brewers",
    "ロサンゼルス・エンゼルス": "Los Angeles Angels",
    "ロサンゼルス・ドジャース": "Los Angeles Dodgers",
    "ワシントン・ナショナルズ": "Washington Nationals",
    # Short names used in FA/retirement entries
    "オリオールズ": "Orioles",
    "ヤンキース": "Yankees",
    "ドジャース": "Dodgers",
}


def translate_team(team_ja: str) -> str:
    """Translate Japanese MLB team name to English."""
    if team_ja in MLB_TEAM_MAP:
        return MLB_TEAM_MAP[team_ja]
    if "FA" in team_ja:
        m = re.search(r"前(.+?)[）)]", team_ja)
        if m:
            prev = MLB_TEAM_MAP.get(m.group(1), m.group(1))
            return f"FA (former {prev})"
        return "Free Agent"
    if "引退復帰" in team_ja:
        m = re.search(r"前(.+?)[）)]", team_ja)
        if m:
            prev = MLB_TEAM_MAP.get(m.group(1), m.group(1))
            return f"Returning from retirement (former {prev})"
        return "Returning from retirement"
    return team_ja


# Roster CSV country header patterns
ROSTER_COUNTRY_MAP = {
    "USA": "USA",
    "Japan": "JPN",
    "Dominican Rep.": "DOM",
    "Venezuela": "VEN",
    "Puerto Rico": "PUR",
    "Mexico": "MEX",
    "Korea": "KOR",
    "Chinese Taipei": "TPE",
    "Netherlands": "NED",
    "Cuba": "CUB",
    "Canada": "CAN",
    "Italy": "ITA",
    "Israel": "ISR",
    "Great Britain": "GBR",
    "Panama": "PAN",
    "Colombia": "COL",
    "Nicaragua": "NCA",
    "Australia": "AUS",
    "Brazil": "BRA",
    "Czechia": "CZE",
}


# ---------------------------------------------------------------------------
# 1. Parse roster CSV
# ---------------------------------------------------------------------------
def parse_roster() -> pd.DataFrame:
    """Parse the semi-structured wbc2026_rosters.csv into a clean table."""
    raw = (DATA_DIR / "wbc2026_rosters.csv").read_text(encoding="utf-8")
    lines = raw.strip().split("\n")

    rows = []
    current_country = None
    current_pool = None

    country_pattern = re.compile(
        r"^(.+?)（.+?）\s*—\s*プール([A-D])（(.+?)）\s*—"
    )

    for line in lines:
        line = line.strip()
        if not line or line.startswith("WBC 2026") or line.startswith("出典"):
            continue

        # Country header line
        m = country_pattern.match(line)
        if m:
            country_name = m.group(1).strip()
            pool_letter = m.group(2)
            pool_city = m.group(3)
            current_country = ROSTER_COUNTRY_MAP.get(country_name)
            city_en = POOL_CITY_MAP.get(pool_city, pool_city)
            current_pool = f"{pool_letter} ({city_en})"
            continue

        # Skip column header line
        if line.startswith("選手名,"):
            continue

        # Player line
        parts = line.split(",")
        if len(parts) >= 3 and current_country and parts[0].strip():
            name = parts[0].strip()
            pos = parts[1].strip() if len(parts) > 1 else ""
            team = translate_team(parts[2].strip()) if len(parts) > 2 else ""
            on_40man = parts[3].strip() if len(parts) > 3 else ""

            # Clean up name: remove Japanese portion "漢字 / English"
            if " / " in name:
                # "大谷翔平 / Shohei Ohtani" -> keep both
                name_ja, name_en = name.split(" / ", 1)
            elif "（" in name:
                # "Woo Suk Go（高宇錫）" -> keep English
                name_en = re.sub(r"（.+?）", "", name).strip()
                name_ja = ""
            else:
                name_en = name
                name_ja = ""

            # Determine role from position
            role = "pitcher" if pos in ("RHP", "LHP") else "batter"

            rows.append({
                "name": name_en,
                "name_ja": name_ja,
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
def combine_statcast(pattern_suffix: str, id_col: str,
                     exclude_pattern: str = "") -> pd.DataFrame:
    """Combine all Statcast CSVs matching pattern into one DataFrame."""
    frames = []
    for csv_path in sorted(DATA_DIR.glob(f"*{pattern_suffix}")):
        # Skip files matching exclude pattern (e.g. pitchers when collecting batters)
        if exclude_pattern and exclude_pattern in csv_path.name:
            continue
        # Extract country prefix from filename
        stem = csv_path.stem  # e.g. "dr_statcast" or "dr_pitchers_statcast"
        prefix = stem.replace(pattern_suffix.replace(".csv", ""), "").rstrip("_")
        if prefix not in COUNTRY_MAP:
            print(f"  SKIP unknown prefix: {prefix} ({csv_path.name})")
            continue

        iso, country_name, pool = COUNTRY_MAP[prefix]
        df = pd.read_csv(csv_path)
        df["country"] = iso
        df["country_name"] = country_name
        frames.append(df)
        print(f"  {csv_path.name}: {len(df):,} rows ({iso})")

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# 3. Compute summary stats
# ---------------------------------------------------------------------------
def batting_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player batting summary from Statcast event-level data."""
    ab_events = {
        "single", "double", "triple", "home_run", "field_out",
        "strikeout", "grounded_into_double_play", "double_play",
        "force_out", "fielders_choice", "fielders_choice_out",
        "strikeout_double_play", "triple_play", "field_error",
    }
    pa_events = ab_events | {
        "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
        "sac_fly_double_play", "catcher_interf", "intent_walk",
    }
    hit_events = {"single", "double", "triple", "home_run"}

    rows = []
    for (batter_id, country), gdf in df.groupby(["batter", "country"]):
        name = gdf["player_name"].iloc[0]
        evts = gdf.dropna(subset=["events"])
        pa = evts[evts["events"].isin(pa_events)]
        ab = evts[evts["events"].isin(ab_events)]
        hits = evts[evts["events"].isin(hit_events)]

        n_pa = len(pa)
        n_ab = len(ab)
        n_h = len(hits)
        n_bb = len(pa[pa["events"].isin({"walk", "intent_walk"})])
        n_hbp = len(pa[pa["events"].isin({"hit_by_pitch"})])
        n_1b = len(hits[hits["events"] == "single"])
        n_2b = len(hits[hits["events"] == "double"])
        n_3b = len(hits[hits["events"] == "triple"])
        n_hr = len(hits[hits["events"] == "home_run"])
        n_k = len(ab[ab["events"].str.contains("strikeout", na=False)])
        tb = n_1b + 2 * n_2b + 3 * n_3b + 4 * n_hr

        avg = n_h / n_ab if n_ab else np.nan
        obp = (n_h + n_bb + n_hbp) / n_pa if n_pa else np.nan
        slg = tb / n_ab if n_ab else np.nan
        ops = obp + slg if not (pd.isna(obp) or pd.isna(slg)) else np.nan
        k_pct = n_k / n_pa * 100 if n_pa else np.nan
        bb_pct = n_bb / n_pa * 100 if n_pa else np.nan
        xwoba = gdf["estimated_woba_using_speedangle"].dropna().mean()
        avg_ev = gdf["launch_speed"].dropna().mean()
        avg_la = gdf["launch_angle"].dropna().mean()

        rows.append({
            "mlbam_id": int(batter_id),
            "player_name": name,
            "country": country,
            "PA": n_pa,
            "AB": n_ab,
            "H": n_h,
            "1B": n_1b,
            "2B": n_2b,
            "3B": n_3b,
            "HR": n_hr,
            "BB": n_bb,
            "HBP": n_hbp,
            "K": n_k,
            "TB": tb,
            "AVG": round(avg, 3) if not pd.isna(avg) else np.nan,
            "OBP": round(obp, 3) if not pd.isna(obp) else np.nan,
            "SLG": round(slg, 3) if not pd.isna(slg) else np.nan,
            "OPS": round(ops, 3) if not pd.isna(ops) else np.nan,
            "K_pct": round(k_pct, 1) if not pd.isna(k_pct) else np.nan,
            "BB_pct": round(bb_pct, 1) if not pd.isna(bb_pct) else np.nan,
            "xwOBA": round(xwoba, 3) if not pd.isna(xwoba) else np.nan,
            "avg_exit_velo": round(avg_ev, 1) if not pd.isna(avg_ev) else np.nan,
            "avg_launch_angle": round(avg_la, 1) if not pd.isna(avg_la) else np.nan,
        })

    return pd.DataFrame(rows).sort_values(["country", "player_name"]).reset_index(drop=True)


def pitching_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-player pitching summary from Statcast pitch-level data."""
    ab_events = {
        "single", "double", "triple", "home_run", "field_out",
        "strikeout", "grounded_into_double_play", "double_play",
        "force_out", "fielders_choice", "fielders_choice_out",
        "strikeout_double_play", "triple_play", "field_error",
    }
    pa_events = ab_events | {
        "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
        "sac_fly_double_play", "catcher_interf", "intent_walk",
    }
    hit_events = {"single", "double", "triple", "home_run"}

    rows = []
    for (pitcher_id, country), gdf in df.groupby(["pitcher", "country"]):
        name = gdf["player_name"].iloc[0]
        total_pitches = len(gdf)
        evts = gdf.dropna(subset=["events"])
        pa = evts[evts["events"].isin(pa_events)]
        ab = evts[evts["events"].isin(ab_events)]
        hits = evts[evts["events"].isin(hit_events)]

        n_pa = len(pa)
        n_ab = len(ab)
        n_h = len(hits)
        n_k = len(ab[ab["events"].str.contains("strikeout", na=False)])
        n_bb = len(pa[pa["events"].isin({"walk", "intent_walk"})])
        n_hr = len(hits[hits["events"] == "home_run"])
        n_1b = len(hits[hits["events"] == "single"])
        n_2b = len(hits[hits["events"] == "double"])
        n_3b = len(hits[hits["events"] == "triple"])
        tb = n_1b + 2 * n_2b + 3 * n_3b + 4 * n_hr

        opp_avg = n_h / n_ab if n_ab else np.nan
        opp_slg = tb / n_ab if n_ab else np.nan
        k_pct = n_k / n_pa * 100 if n_pa else np.nan
        bb_pct = n_bb / n_pa * 100 if n_pa else np.nan
        xwoba = gdf["estimated_woba_using_speedangle"].dropna().mean()
        avg_velo = gdf["release_speed"].dropna().mean() if "release_speed" in gdf.columns else np.nan
        avg_spin = gdf["release_spin_rate"].dropna().mean() if "release_spin_rate" in gdf.columns else np.nan

        # Pitch type distribution
        pitch_counts = gdf["pitch_type"].value_counts()
        top_pitch = pitch_counts.index[0] if len(pitch_counts) > 0 else ""
        n_pitch_types = len(pitch_counts)

        rows.append({
            "mlbam_id": int(pitcher_id),
            "player_name": name,
            "country": country,
            "total_pitches": total_pitches,
            "PA_faced": n_pa,
            "K": n_k,
            "BB": n_bb,
            "HR_allowed": n_hr,
            "H_allowed": n_h,
            "opp_AVG": round(opp_avg, 3) if not pd.isna(opp_avg) else np.nan,
            "opp_SLG": round(opp_slg, 3) if not pd.isna(opp_slg) else np.nan,
            "K_pct": round(k_pct, 1) if not pd.isna(k_pct) else np.nan,
            "BB_pct": round(bb_pct, 1) if not pd.isna(bb_pct) else np.nan,
            "xwOBA_against": round(xwoba, 3) if not pd.isna(xwoba) else np.nan,
            "avg_velo": round(avg_velo, 1) if not pd.isna(avg_velo) else np.nan,
            "avg_spin_rate": round(avg_spin, 0) if not pd.isna(avg_spin) else np.nan,
            "pitch_type_count": n_pitch_types,
            "primary_pitch": top_pitch,
        })

    return pd.DataFrame(rows).sort_values(["country", "player_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== WBC 2026 Kaggle Dataset Generator ===\n")

    # 1. Roster
    print("[1/6] Parsing roster...")
    roster = parse_roster()
    roster.to_csv(OUT_DIR / "rosters.csv", index=False)
    print(f"  -> rosters.csv: {len(roster)} players, "
          f"{roster['country'].nunique()} countries\n")

    # 2. Combine batter Statcast
    print("[2/6] Combining batter Statcast CSVs...")
    # Batter CSVs: *_statcast.csv but NOT *_pitchers_statcast.csv
    batter_raw = combine_statcast("_statcast.csv", "batter",
                                  exclude_pattern="_pitchers_")
    # Filter out pitcher CSVs that were picked up
    if not batter_raw.empty:
        batter_raw.to_csv(OUT_DIR / "statcast_batters.csv", index=False)
        print(f"  -> statcast_batters.csv: {len(batter_raw):,} rows\n")

    # 3. Combine pitcher Statcast
    print("[3/6] Combining pitcher Statcast CSVs...")
    pitcher_raw = combine_statcast("_pitchers_statcast.csv", "pitcher")
    if not pitcher_raw.empty:
        pitcher_raw.to_csv(OUT_DIR / "statcast_pitchers.csv", index=False)
        print(f"  -> statcast_pitchers.csv: {len(pitcher_raw):,} rows\n")

    # 4. Batter summary
    print("[4/6] Computing batter summary stats...")
    if not batter_raw.empty:
        bat_summary = batting_summary(batter_raw)
        bat_summary.to_csv(OUT_DIR / "batter_summary.csv", index=False)
        print(f"  -> batter_summary.csv: {len(bat_summary)} players\n")

    # 5. Pitcher summary
    print("[5/6] Computing pitcher summary stats...")
    if not pitcher_raw.empty:
        pit_summary = pitching_summary(pitcher_raw)
        pit_summary.to_csv(OUT_DIR / "pitcher_summary.csv", index=False)
        print(f"  -> pitcher_summary.csv: {len(pit_summary)} players\n")

    # 6. Stadiums
    print("[6/6] Copying stadium data...")
    stadiums_src = DATA_DIR / "mlbstadiums_wbc.csv"
    if stadiums_src.exists():
        stadiums = pd.read_csv(stadiums_src)
        # Clean up unnamed index column
        if "Unnamed: 0" in stadiums.columns:
            stadiums = stadiums.drop(columns=["Unnamed: 0"])
        stadiums.to_csv(OUT_DIR / "stadiums.csv", index=False)
        print(f"  -> stadiums.csv: {len(stadiums):,} rows\n")

    print("=== Done! ===")


if __name__ == "__main__":
    main()
