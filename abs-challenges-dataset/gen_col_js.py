"""Write col_<file>.js for the two ABS CSVs.

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
for sfx, where in (("aaa2025", "Triple-A 2025 regular season"), ("mlb2026", "MLB 2026 regular season")):
    for c in BRIDGE_COLS:
        BRIDGE[f"{c}_{sfx}"] = f"{where}: {BASE[c]}" + (" (column names refer to abs_challenges_players.csv)" if c in FORMULA_COLS else "")

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


if __name__ == "__main__":
    main()
