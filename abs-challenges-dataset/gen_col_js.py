"""Write col_<file>.js and settings.json for the two ABS CSVs.

settings.json is what the "Update Kaggle Metadata" workflow sends (scripts/update_metadata.py).

Paste a generated file into the browser console on the Kaggle dataset's column-description editor
(Data tab, edit the file's columns) to fill every column's description, then click Save. Same pattern
as savant-extras-dataset/col_*.js. Only the CSVs, file_descriptions.txt and dataset-metadata.json are
uploaded (build.py copies *.txt only), so these files do not change the dataset itself.

Every formula stated below was checked on every row of the built CSVs where both sides are present.
Where Savant's own formula could not be reproduced, the description says so instead of guessing.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

BASE = {
    "challenge_type": "Who challenged on this board: batter, pitcher or catcher.",
    "player_id": "MLB Advanced Media (MLBAM) player id. Joins to Statcast batter / pitcher / fielder_2.",
    "player_name": "Player name as shown by Baseball Savant.",
    "team_abbr": "Team abbreviation at the board's level (the Triple-A club on AAA boards, the MLB club on MLB boards); one team is shown when n_teams > 1.",
    "parent_org": "Abbreviation of the MLB parent organisation (equals team_abbr on MLB boards).",
    "level": "MLB or AAA (Triple-A).",
    "year": "Season (2025 or 2026).",
    "game_type": "Added by this dataset: R = regular season, S = spring training.",
    "n_total_sample": "Savant's sample size for the player on this board; the denominator of rate_challenges.",
    "exp_chal": "Savant's expected challenges = exp_chal_gained + exp_chal_lost.",
    "exp_chal_gained": "Savant's expected overturned (won) challenges.",
    "exp_chal_lost": "Savant's expected failed (lost) challenges.",
    "exp_chal_runs_gained": "Savant's expected run value from overturned challenges.",
    "exp_chal_runs_lost": "Savant's expected run value lost on failed challenges.",
    "n_challenges": "Challenges made.",
    "n_overturns": "Challenges won (the call was overturned).",
    "net_chal_gained": "n_overturns - exp_chal_gained.",
    "net_chal_lost": "n_fails - exp_chal_lost.",
    "n_chal_reasonable": "Challenges Savant classes as reasonable.",
    "n_chal_reasonable_opps": "Savant's count of reasonable challenge opportunities.",
    "n_fails": "Challenges lost = n_challenges - n_overturns.",
    "n_chal_runs": "Run value of the challenges = n_chal_runs_gained - n_chal_runs_lost.",
    "n_chal_runs_gained": "Savant's run value gained on overturned challenges (never negative).",
    "n_chal_runs_lost": "Savant's run value lost on failed challenges (never negative).",
    "net_chal_gained_runs": "Savant's net runs on overturned challenges vs expected. Equals n_chal_runs_gained - exp_chal_runs_gained on batter boards; on most pitcher and catcher rows it does not, and Savant's formula there is not reproduced here.",
    "net_chal_lost_runs": "Savant's net runs on failed challenges vs expected. Equals n_chal_runs_lost - exp_chal_runs_lost on batter boards; on most pitcher and catcher rows it does not, and Savant's formula there is not reproduced here.",
    "net_net_runs": "net_chal_gained_runs - net_chal_lost_runs.",
    "n_strikeouts": "Savant's strikeout count on the board.",
    "n_walks": "Savant's walk count on the board.",
    "player_team": "Savant's numeric team id for team_abbr (equals parent_org_code on MLB boards).",
    "n_teams": "Number of teams the player appeared for on this board (1-4).",
    "n_levels": "Number of levels on this board (always 1).",
    "player_at_bat": "Equals player_id; filled on batter boards only.",
    "parent_org_code": "Numeric MLB team id of the parent organisation.",
    "levelCode": "Savant's raw level code: \"1\" on MLB boards, the string \"Triple-A\" on AAA boards.",
    "rate_challenges": "n_challenges / n_total_sample.",
    "exp_rate_challenges": "exp_chal / n_total_sample.",
    "exp_rate_challenges_diff": "rate_challenges - exp_rate_challenges.",
    "rate_overturns": "n_overturns / n_challenges (empty when there are no challenges).",
    "exp_rate_overturns": "exp_chal_gained / exp_chal (empty when exp_chal_gained is 0).",
    "net_net_chal": "net_chal_gained - net_chal_lost.",
    "exp_chal_runs": "exp_chal_runs_gained - exp_chal_runs_lost.",
    "rate_chal_reasonable": "n_chal_reasonable / n_challenges (empty when there are no challenges).",
    "rate_reasonable_opp_taken": "n_chal_reasonable / n_chal_reasonable_opps (empty when n_chal_reasonable_opps is 0).",
    "runs_gained_per_chal": "n_chal_runs_gained / n_challenges (empty when there are no challenges).",
    "exp_runs_gained_per_chal": "exp_chal_runs_gained / exp_chal.",
    "overturns_vs_exp": "net_net_chal - net_net_chal_against: the player's own net overturns vs expected, minus the same for challenges made against the player's side.",
    "runs_vs_exp": "net_net_runs - net_net_runs_against.",
    "net_net_chal_against_proxy": "Equals -net_net_chal_against.",
    "net_chal_gained_runs_against_proxy": "Equals -net_chal_gained_runs_against.",
    "net_net_runs_against_proxy": "Equals -net_net_runs_against.",
    "uniqueId": "Savant's row key: player_id and year joined by an underscore.",
    "pitcher": "Equals player_id; filled on pitcher boards only.",
    "pitcher_against": "Equals pitcher where filled; pitcher boards only.",
    "fielder_2": "Equals player_id; filled on catcher boards only (Statcast's catcher column).",
    "fielder_2_against": "Equals fielder_2 where filled; catcher boards only.",
    "year_1_against": "Season of the _against measures (equals year).",
}

# Columns that also appear with an _against suffix: the same measure for challenges made against
# the player's side. Their formulas were checked with every name suffixed.
AGAINST = [
    "n_total_sample", "exp_chal", "exp_chal_gained", "exp_chal_lost", "exp_chal_runs_gained",
    "exp_chal_runs_lost", "n_challenges", "n_overturns", "net_chal_gained", "net_chal_lost",
    "n_chal_reasonable", "n_chal_reasonable_opps", "n_fails", "n_chal_runs", "n_chal_runs_gained",
    "n_chal_runs_lost", "net_chal_gained_runs", "net_chal_lost_runs", "n_strikeouts", "n_walks",
    "n_teams", "n_levels", "rate_challenges", "exp_rate_challenges", "exp_rate_challenges_diff",
    "rate_overturns", "exp_rate_overturns", "net_net_chal", "exp_chal_runs", "net_net_runs",
    "rate_chal_reasonable", "rate_reasonable_opp_taken", "runs_gained_per_chal", "exp_runs_gained_per_chal",
]
AGAINST_EXACT = {
    # On the _against side the simple formula holds on every board, not only batter boards.
    "net_chal_gained_runs": "n_chal_runs_gained_against - exp_chal_runs_gained_against",
    "net_chal_lost_runs": "n_chal_runs_lost_against - exp_chal_runs_lost_against",
}
for c in AGAINST:
    if c in AGAINST_EXACT:
        text = f"Net runs vs expected for challenges made against the player's side = {AGAINST_EXACT[c]}."
    else:
        text = f"As {c}, but for challenges made against the player by the opposing side (pitchers/catchers for a batter, batters for a pitcher/catcher). Empty when Savant has no such row"
        text += "." if "(empty when" not in BASE[c] else "; also " + BASE[c][BASE[c].index("(empty when") + 1:-2].replace("_gained", "_gained_against").replace("opps is", "opps_against is").replace("no challenges", "no such challenges") + "."
    BASE[c + "_against"] = text

# Columns added by enrich.py (player bio, season stats and Statcast aggregates for the same level,
# season and game type as the row's board). Formulas stated here are computed by enrich.py itself.
LVL = "for the row's level, season and game type"
BIO = {
    "bats": "Batting side from MLB StatsAPI: R, L or S (switch).",
    "throws": "Throwing hand from MLB StatsAPI: R or L.",
    "birth_date": "Date of birth (YYYY-MM-DD) from MLB StatsAPI.",
    "age": "Age on June 30 of the row's season, from birth_date.",
    "height_in": "Listed height in inches (MLB StatsAPI).",
    "weight_lb": "Listed weight in pounds (MLB StatsAPI).",
    "primary_position": "Primary position abbreviation from MLB StatsAPI (P, C, 1B, 2B, 3B, SS, LF, CF, RF, OF, DH, ...).",
    "mlb_debut_date": "MLB debut date from MLB StatsAPI; empty if the player has not debuted.",
    "birth_country": "Country of birth from MLB StatsAPI.",
}
API_BAT = {
    "g": "games", "pa": "plate appearances", "ab": "at-bats", "h": "hits", "2b": "doubles",
    "3b": "triples", "hr": "home runs", "so": "strikeouts", "bb": "walks", "hbp": "hit by pitch",
    "sb": "stolen bases", "avg": "batting average", "obp": "on-base percentage",
    "slg": "slugging percentage", "ops": "OBP + SLG", "pitches": "pitches seen",
}
API_PIT = {
    "g": "games pitched", "gs": "games started", "outs": "outs recorded", "bf": "batters faced",
    "so": "strikeouts", "bb": "walks", "hbp": "hit batters", "hr": "home runs allowed", "era": "ERA",
    "whip": "WHIP", "strike_pct": "share of pitches that were strikes, as a fraction 0-1",
    "pitches": "pitches thrown", "wp": "wild pitches", "ip": "innings pitched as a decimal (outs / 3)",
}
API_C = {
    "g": "games at catcher", "innings": "innings at catcher as a decimal (12.1 in box-score notation = 12.333)",
    "sb": "stolen bases allowed", "cs": "runners caught stealing", "cs_pct": "caught-stealing share, 0-1",
    "pb": "passed balls", "wp": "wild pitches while catching", "catcher_era": "ERA of pitchers while this player caught",
}
SC_COMMON = {
    "pitches": "pitches", "pa": "plate appearances", "k_percent": "strikeout rate, %",
    "bb_percent": "walk rate, %", "swings": "swings", "takes": "takes (pitches not swung at)",
    "whiffs": "whiffs (swinging strikes and foul tips)", "swing_miss_percent": "whiffs / swings, %", "woba": "wOBA",
    "xwoba": "expected wOBA", "xba": "expected batting average", "xslg": "expected slugging",
    "launch_speed": "mean exit velocity of batted balls, mph", "launch_angle": "mean launch angle, degrees",
    "hardhit_percent": "share of batted balls at 95+ mph, %", "barrels_per_bbe_percent": "barrels per batted ball, %",
    "run_exp": "Savant's run_exp summed over the pitches (Statcast run-value change; positive is good for the batting side)",
    "oz_pitches": "pitches outside the strike zone (Statcast zones 11-14)",
    "oz_swings": "swings at pitches outside the zone", "oz_whiffs": "whiffs on pitches outside the zone",
    "iz_pitches": "pitches inside the strike zone (Statcast zones 1-9)",
    "iz_swings": "swings at pitches inside the zone", "iz_whiffs": "whiffs on pitches inside the zone",
    "zone_tracked_percent": "(oz_pitches + iz_pitches) / pitches, %: share of pitches with a Statcast zone (about 2/3 in MLB 2025 spring, 99%+ elsewhere)",
    "chase_percent": "oz_swings / oz_pitches, %", "zone_swing_percent": "iz_swings / iz_pitches, %",
    "zone_contact_percent": "(1 - iz_whiffs / iz_swings), %", "chase_contact_percent": "(1 - oz_whiffs / oz_swings), %",
}
SC_C = {
    "pitches_received": "pitches caught (Statcast fielder_2)", "takes_received": "takes on those pitches",
    "oz_takes": "takes outside the zone (zones 11-14)", "oz_called_strikes": "of those, called strikes",
    "iz_takes": "takes inside the zone (zones 1-9)", "iz_called_balls": "of those, called balls",
    "oz_called_strike_percent": "oz_called_strikes / oz_takes, %",
    "iz_called_ball_percent": "iz_called_balls / iz_takes, %",
    "zone_tracked_percent": "(oz_takes + iz_takes) / takes_received, %: share of takes with a Statcast zone",
}
NEW = dict(BIO)
for k, v in API_BAT.items():
    NEW[f"api_bat_{k}"] = f"MLB StatsAPI hitting {LVL}: {v}. Batter and catcher boards."
for k, v in API_PIT.items():
    NEW[f"api_pit_{k}"] = f"MLB StatsAPI pitching {LVL}: {v}. Pitcher boards."
for k, v in API_C.items():
    NEW[f"api_c_{k}"] = f"MLB StatsAPI fielding at C {LVL}: {v}. Catcher boards."
for k, v in SC_COMMON.items():
    NEW[f"sc_bat_{k}"] = f"Statcast search {LVL}, as the batter: {v}. Batter boards."
    NEW[f"sc_pit_{k}"] = f"Statcast search {LVL}, against the pitcher: {v}. Pitcher boards."
for k, v in SC_C.items():
    NEW[f"sc_c_{k}"] = f"Statcast search {LVL}, as the catcher: {v}. Catcher boards; zone calls as recorded by Statcast."
NEW.update({
    "api_bat_woba": "MLB 2026 regular season only: wOBA from MLB StatsAPI sabermetrics.",
    "api_bat_wrc_plus": "MLB 2026 regular season only: wRC+ from MLB StatsAPI sabermetrics.",
    "api_bat_war": "MLB 2026 regular season only: batter WAR from MLB StatsAPI sabermetrics.",
    "api_pit_war": "MLB 2026 regular season only: pitcher WAR from MLB StatsAPI sabermetrics.",
    "api_pit_fip": "MLB 2026 regular season only: FIP from MLB StatsAPI sabermetrics.",
    "api_pit_xfip": "MLB 2026 regular season only: xFIP from MLB StatsAPI sabermetrics.",
    "sc_c_framing_runs": "MLB 2026 regular season only: framing runs (rv_tot) from Savant's catcher-framing leaderboard (qualified catchers only; empty for others).",
    "sc_c_framing_strike_rate": "MLB 2026 regular season only: pct_tot from Savant's catcher-framing leaderboard (strike rate on the pitches it scores), 0-1.",
})
BASE.update(NEW)

BRIDGE_COLS = [
    "team_abbr", "parent_org", "n_total_sample", "n_challenges", "n_overturns", "n_fails",
    "rate_challenges", "rate_overturns", "exp_chal", "exp_rate_overturns", "overturns_vs_exp",
    "n_chal_runs", "exp_chal_runs", "runs_vs_exp", "n_chal_reasonable_opps",
    "rate_reasonable_opp_taken", "n_strikeouts", "n_walks",
]
FORMULA_COLS = {"n_fails", "rate_challenges", "rate_overturns", "exp_chal", "exp_rate_overturns", "overturns_vs_exp", "n_chal_runs", "exp_chal_runs", "runs_vs_exp", "rate_reasonable_opp_taken"}
BRIDGE = {
    "challenge_type": "batter or catcher (pitchers are not in this table).",
    "player_id": BASE["player_id"],
    "player_name": BASE["player_name"],
}
for c in BIO:
    if c != "age":  # age depends on the season: age_aaa2025 / age_mlb2026
        BRIDGE[c] = BIO[c]
# Only on MLB 2026 regular-season rows, so only on the _mlb2026 side of the bridge.
MLB2026_ONLY = {"api_bat_woba", "api_bat_wrc_plus", "api_bat_war", "sc_c_framing_runs", "sc_c_framing_strike_rate"}
BRIDGE_ENRICH = ["age"] + [c for c in NEW if c.startswith(("api_bat_", "api_c_", "sc_bat_", "sc_c_"))]
for sfx, where in (("aaa2025", "Triple-A 2025 regular season"), ("mlb2026", "MLB 2026 regular season")):
    for c in BRIDGE_COLS:
        BRIDGE[f"{c}_{sfx}"] = f"{where}: {BASE[c]}" + (" (column names refer to abs_challenges_players.csv)" if c in FORMULA_COLS else "")
    for c in BRIDGE_ENRICH:
        if sfx == "aaa2025" and c in MLB2026_ONLY:
            continue
        text = NEW[c].replace(" " + LVL, "").replace("MLB 2026 regular season only: ", "")
        BRIDGE[f"{c}_{sfx}"] = f"{where}: {text}"

TEMPLATE = HERE.parent / "savant-extras-dataset" / "col_arm_strength.js"


def main() -> None:
    import csv
    body = TEMPLATE.read_text(encoding="utf-8")
    tail = body[body.index("    let updated"):]
    targets = (("abs_challenges_players", BASE), ("abs_aaa2025_to_mlb2026", BRIDGE))
    for name, cols in targets:
        csv_path = HERE / "_build" / f"{name}.csv"
        if csv_path.exists():  # when a build is present, the dict must cover exactly its columns
            header = next(csv.reader(csv_path.open(encoding="utf-8")))
            missing, extra = set(header) - set(cols), set(cols) - set(header)
            assert not missing and not extra, (name, sorted(missing), sorted(extra))
        for k, v in cols.items():
            assert len(v) <= 250, (k, len(v))
        inner = ",\n".join(f"        {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in cols.items())
        js = "(function() {\n    const columns = {\n" + inner + "\n    };\n" + tail
        (HERE / f"col_{name}.js").write_text(js, encoding="utf-8")
        print(f"col_{name}.js: {len(cols)} columns")
    write_settings(dict(targets))


SOURCES = (
    "Baseball Savant ABS challenge leaderboard (MLB Advanced Media): "
    "https://baseballsavant.mlb.com/leaderboard/abs-challenges . "
    "Player columns: MLB StatsAPI (https://statsapi.mlb.com, /people and season stats) and "
    "Baseball Savant Statcast search (https://baseballsavant.mlb.com/statcast_search) and catcher-framing leaderboard. "
    "Collected with savant-extras 0.6.0 (https://pypi.org/project/savant-extras/) by build.py in "
    "https://github.com/yasumorishima/kaggle-datasets (abs-challenges-dataset), which fails without "
    "writing anything unless every gate in the dataset description passes."
)


def write_settings(cols_by_file: dict) -> None:
    """settings.json for scripts/update_metadata.py: file descriptions from file_descriptions.txt,
    column descriptions from the dicts above, update frequency and sources."""
    import re
    text = (HERE / "file_descriptions.txt").read_text(encoding="utf-8")
    parts = re.split(r"-{20,}\n(\S+\.csv)\n-{20,}\n", text)
    fdesc = {parts[i]: " ".join(parts[i + 1].split()) for i in range(1, len(parts), 2)}
    files = {}
    for name, cols in cols_by_file.items():
        fname = f"{name}.csv"
        assert fdesc.get(fname), fname
        files[fname] = {"description": fdesc[fname], "columns": cols}
    assert set(files) == set(fdesc), (sorted(files), sorted(fdesc))
    meta = json.loads((HERE / "dataset-metadata.json").read_text(encoding="utf-8"))
    settings = {
        "id": meta["id"],
        "expectedUpdateFrequency": "never",  # user decision 2026-09-27: fixed after the post-season version
        "userSpecifiedSources": SOURCES,
        "files": files,
    }
    (HERE / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"settings.json: {len(files)} files")


if __name__ == "__main__":
    main()
