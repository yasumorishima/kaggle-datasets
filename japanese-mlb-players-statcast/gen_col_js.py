"""Write col_<file>.js and settings.json for the four CSVs of this dataset.

settings.json is what the "Update Kaggle Metadata" workflow sends (scripts/update_metadata.py);
build.py also refuses to build when a column of an output file has no description here.

Paste a generated col_*.js into the browser console on the Kaggle dataset's column-description
editor (Data tab, edit the file's columns) to fill every column's description, then click Save
(the public API does not return per-file columns). Same pattern as abs-challenges-dataset.

The Statcast descriptions start from the earlier version's (dataset1_japanese_mlb/kaggle_*.js) and
replace the ones that were wrong: pfx_x / pfx_z and the api_break columns are in feet, not inches;
player_name is 'Last, First'; game_type is now always R; and several are rewritten to say only what
was checked on the data (see CHECKED below).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "dataset1_japanese_mlb"
PITCHING, BATTING = "japanese_mlb_pitching.csv", "japanese_mlb_batting.csv"


def old_columns(js: str) -> dict:
    s = (OLD / js).read_text(encoding="utf-8")
    a = s.index("const columns = ") + len("const columns = ")
    return json.loads(s[a:s.index("};", a) + 1])


EMPTY = "Deprecated Savant column, kept so the files have Savant's full schema; empty in every row (the build fails otherwise)."
BAT_TRACKING = "; empty on pitches without a tracked swing and before 2023 (Statcast bat tracking)"
# Replacements that hold for both pitch files. Statements about units, signs and which rows are
# filled were checked on the built data (2026 for every player, 2016 for Iwakuma) and by the build.
COMMON = {
    "game_type": "Game type. Always R (regular season) in this dataset; spring training and postseason pitches are not included.",
    "game_year": "Season of the game (2015-2026).",
    "pfx_x": "Horizontal movement in feet from the catcher's perspective, without gravity (Statcast's induced movement).",
    "pfx_z": "Vertical movement in feet without gravity (induced vertical break): positive = the pitch stays above the path of a spinless pitch.",
    "api_break_z_with_gravity": "Vertical break including gravity, in feet; positive = downward.",
    "api_break_x_arm": "Horizontal break in feet toward the pitcher's arm side (positive = arm side).",
    "api_break_x_batter_in": "Horizontal break in feet toward the batter (positive = in on the batter).",
    "launch_speed_angle": "Statcast launch speed/angle category of a batted ball: 1 Weak, 2 Topped, 3 Under, 4 Flare/Burner, 5 Solid Contact, 6 Barrel.",
    "babip_value": "BABIP value Statcast assigns to the plate-appearance outcome (1 = hit on a ball in play other than a home run).",
    "iso_value": "Isolated-power value of the outcome (extra bases: 1 double, 2 triple, 3 home run; 0 otherwise).",
    "hyper_speed": "Statcast's hyper speed: launch_speed with a floor of 88 mph (launch_speed at 88 and above, 88 below). Where launch_speed is empty it is empty or 88 (some foul bunts).",
    "miss_distance": "Statcast's miss distance on a swinging strike (how far the bat missed the ball); empty on other pitches and before 2023. Savant does not state the unit.",
    "bat_speed": "Bat speed in mph from Statcast bat tracking" + BAT_TRACKING + ".",
    "swing_length": "Swing length in feet from Statcast bat tracking" + BAT_TRACKING + ".",
    "attack_angle": "Bat's vertical attack angle at contact in degrees (Statcast bat tracking)" + BAT_TRACKING + ".",
    "attack_direction": "Bat's horizontal attack direction in degrees (Statcast bat tracking)" + BAT_TRACKING + ".",
    "swing_path_tilt": "Tilt of the swing plane in degrees (Statcast bat tracking)" + BAT_TRACKING + ".",
    "intercept_ball_minus_batter_pos_x_inches": "Statcast bat tracking: horizontal distance in inches between where bat meets ball and the batter's position" + BAT_TRACKING + ".",
    "intercept_ball_minus_batter_pos_y_inches": "Statcast bat tracking: depth distance in inches between where bat meets ball and the batter's position" + BAT_TRACKING + ".",
    "arm_angle": "Pitcher's arm angle at release in degrees (Savant); mostly filled from 2020 in this dataset, empty in 2015-2019.",
    "age_pit": "Pitcher's age as Savant gives it now. age_pit_legacy is Savant's earlier age field; the two can differ by one year.",
    "age_bat": "Batter's age as Savant gives it now. age_bat_legacy is Savant's earlier age field; the two can differ by one year.",
    "age_pit_legacy": "Savant's earlier pitcher age field (see age_pit).",
    "age_bat_legacy": "Savant's earlier batter age field (see age_bat).",
    "sv_id": "Savant's old pitch id (YYMMDD_hhmmss); filled in 2015-2016 only in this dataset.",
    "umpire": EMPTY,
    "spin_dir": EMPTY,
    "spin_rate_deprecated": EMPTY,
    "break_angle_deprecated": EMPTY,
    "break_length_deprecated": EMPTY,
    "tfs_deprecated": EMPTY,
    "tfs_zulu_deprecated": EMPTY,
    "events": "Plate-appearance outcome (e.g. single, strikeout, home_run, walk); filled only on the last pitch of a plate appearance.",
    "pitch_type": "Pitch type code as classified by Statcast (FF four-seam, SI sinker, FC cutter, SL slider, ST sweeper, CU curveball, CH changeup, FS splitter, ...).",
}
ROLE = {
    PITCHING: {
        "player_name": "The pitcher's name as Baseball Savant writes it ('Last, First'). The build checks it against MLB StatsAPI's name for the pitcher id.",
        "player_name_eng": "Added by this dataset (not a Statcast column): the pitcher's full name from MLB StatsAPI ('First Last').",
        "pitcher": "MLBAM id of the pitcher: the Japanese player of the row. Joins to mlbam_id in players.csv and season_summary.csv (role pitcher).",
        "batter": "MLBAM id of the batter facing the pitch.",
    },
    BATTING: {
        "player_name": "The batter's name as Baseball Savant writes it ('Last, First'). The build checks it against MLB StatsAPI's name for the batter id.",
        "player_name_eng": "Added by this dataset (not a Statcast column): the batter's full name from MLB StatsAPI ('First Last').",
        "batter": "MLBAM id of the batter: the Japanese player of the row. Joins to mlbam_id in players.csv and season_summary.csv (role batter).",
        "pitcher": "MLBAM id of the pitcher throwing the pitch.",
    },
}

PLAYERS = {
    "mlbam_id": "MLB Advanced Media player id. Joins to pitcher / batter in the pitch files and to season_summary.csv.",
    "name": "Full name from MLB StatsAPI.",
    "name_last_first": "'Last, First' from MLB StatsAPI (the form of Savant's player_name).",
    "birth_country": "Birth country from MLB StatsAPI: Japan, except the two players with a heritage_note.",
    "birth_city": "Birth city from MLB StatsAPI.",
    "birth_date": "Birth date from MLB StatsAPI (YYYY-MM-DD).",
    "heritage_note": "'US-born, Japanese heritage' for Lars Nootbaar and Gosuke Katoh, kept from the earlier version by an explicit list; empty otherwise.",
    "primary_position": "Primary position from MLB StatsAPI at build time (P, TWP = two-way player, C, 1B, 2B, 3B, SS, LF, CF, RF, DH).",
    "bats": "Batting side from MLB StatsAPI: L, R or S.",
    "throws": "Throwing hand from MLB StatsAPI: L or R.",
    "mlb_debut_date": "MLB debut date from MLB StatsAPI (YYYY-MM-DD); may be before 2015.",
    "roles": "Which pitch files have rows for the player: pitcher, batter, or both.",
    "teams": "MLB teams the player appeared for in the seasons covered (StatsAPI abbreviation of that season), by season (within a season in StatsAPI's order), separated by /.",
    "seasons": "Comma-separated seasons 2015-2026 with MLB regular-season games.",
    "pitching_seasons": "Seasons with rows in japanese_mlb_pitching.csv (pitches thrown in the regular season).",
    "batting_seasons": "Seasons with rows in japanese_mlb_batting.csv (plate appearances in the regular season, pitchers included).",
    "in_previous_version": "yes if the player was in the earlier (February 2026) version of this dataset, including the two whose id was wrong there; no if new.",
    "previous_wrong_id": "Hisashi Iwakuma and Yuki Matsui only: the wrong id the earlier version used (461325 is Tyler Clippard, 680686 is Josiah Gray). Empty otherwise.",
}

API = "MLB StatsAPI regular-season"
SUMMARY = {
    "mlbam_id": "MLB Advanced Media player id (joins to players.csv and the pitch files).",
    "name": "Full name from MLB StatsAPI.",
    "season": "Season.",
    "role": "pitcher (a row of japanese_mlb_pitching.csv) or batter (japanese_mlb_batting.csv). Two-way seasons have one row per role.",
    "teams": "MLB teams in the season (StatsAPI abbreviations), in StatsAPI's order, separated by /.",
    "games": f"{API} games played (pitching games for role pitcher, batting games for role batter).",
    "statsapi_pitches": f"{API} pitches thrown (pitcher) or seen (batter).",
    "statcast_rows": "Rows of this player-season-role in the pitch file. Minus the automatic_ball / automatic_strike rows it equals statsapi_pitches (the build checks this).",
}
for k, v in {"pa": "plate appearances", "ab": "at-bats", "h": "hits", "hr": "home runs", "bb": "walks",
             "so": "strikeouts", "sb": "stolen bases", "avg": "batting average", "obp": "on-base percentage",
             "slg": "slugging percentage", "ops": "OPS"}.items():
    SUMMARY[f"bat_{k}"] = f"Role batter: {API} {v}. Empty on pitcher rows."
for k, v in {"woba": "wOBA", "wrc_plus": "wRC+", "war": "WAR"}.items():
    SUMMARY[f"bat_{k}"] = f"Role batter: {v} from MLB StatsAPI sabermetrics (MLB's own calculation). Empty on pitcher rows."
for k, v in {"gs": "games started", "ip": "innings pitched (x.1 and x.2 are thirds)", "bf": "batters faced",
             "w": "wins", "l": "losses", "sv": "saves", "hld": "holds", "so": "strikeouts", "bb": "walks",
             "hr": "home runs allowed", "era": "ERA", "whip": "WHIP"}.items():
    SUMMARY[f"pit_{k}"] = f"Role pitcher: {API} {v}. Empty on batter rows."
for k, v in {"fip": "FIP", "xfip": "xFIP", "war": "WAR"}.items():
    SUMMARY[f"pit_{k}"] = f"Role pitcher: {v} from MLB StatsAPI sabermetrics (MLB's own calculation). Empty on batter rows."

SOURCES = (
    "Pitch-by-pitch rows: Baseball Savant Statcast search (MLB Advanced Media), "
    "https://baseballsavant.mlb.com/statcast_search , one CSV per player, season and role, regular season only. "
    "Player list, names, bio and season stats: MLB StatsAPI, https://statsapi.mlb.com (/sports/1/players, /people, "
    "season and sabermetrics stats). Built by build.py in https://github.com/yasumorishima/kaggle-datasets "
    "(japanese-mlb-players-statcast), which fails without writing anything unless every gate in the dataset "
    "description passes."
)

TEMPLATE = HERE.parent / "savant-extras-dataset" / "col_arm_strength.js"


def main() -> None:
    import csv
    import re
    old = {PITCHING: old_columns("kaggle_pitching.js"), BATTING: old_columns("kaggle_batting.js")}
    files = {}
    for fname in (PITCHING, BATTING):
        cols = dict(old[fname])
        cols.update(COMMON)
        cols.update(ROLE[fname])
        files[fname] = cols
    files["players.csv"] = PLAYERS
    files["season_summary.csv"] = SUMMARY

    body = TEMPLATE.read_text(encoding="utf-8")
    tail = body[body.index("    let updated"):]
    for fname, cols in files.items():
        built = HERE / "_build" / fname
        if built.exists():  # with a build present, the dict must cover exactly its columns, in its order
            with built.open(encoding="utf-8") as fh:
                header = next(csv.reader(fh))
            assert set(header) == set(cols), (fname, sorted(set(header) - set(cols)), sorted(set(cols) - set(header)))
            files[fname] = cols = {c: cols[c] for c in header}
        for k, v in cols.items():
            assert len(v) <= 250, (fname, k, len(v))
        inner = ",\n".join(f"        {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in cols.items())
        js = "(function() {\n    const columns = {\n" + inner + "\n    };\n" + tail
        (HERE / f"col_{fname[:-4]}.js").write_text(js, encoding="utf-8")
        print(f"col_{fname[:-4]}.js: {len(cols)} columns")

    text = (HERE / "file_descriptions.txt").read_text(encoding="utf-8")
    parts = re.split(r"-{20,}\n(\S+\.csv)\n-{20,}\n", text)
    fdesc = {parts[i]: " ".join(parts[i + 1].split()) for i in range(1, len(parts), 2)}
    assert set(fdesc) == set(files), (sorted(fdesc), sorted(files))
    meta = json.loads((HERE / "dataset-metadata.json").read_text(encoding="utf-8"))
    settings = {
        "id": meta["id"],
        "expectedUpdateFrequency": "not specified",
        "userSpecifiedSources": SOURCES,
        "files": {n: {"description": fdesc[n], "columns": c} for n, c in files.items()},
    }
    (HERE / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"settings.json: {len(files)} files")


if __name__ == "__main__":
    main()
