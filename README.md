# Kaggle Datasets

Baseball-themed datasets published on Kaggle, generated with [pybaseball](https://github.com/jldbc/pybaseball) and Baseball Savant.

## Published Datasets (7)

### 1. [Japanese MLB Players Statcast (2015-2026)](https://www.kaggle.com/datasets/yasunorim/japan-mlb-pitchers-batters-statcast)

Every regular-season Statcast pitch thrown or seen by Japanese MLB players, 2015-2026.

- **Players:** 38: 28 with pitching seasons, 24 with batting seasons (14 have both: mostly pitchers who batted before the 2022 universal DH, plus Ichiro and Aoki, who pitched, and Ohtani); 153 player-season-roles
- **Files:** `japanese_mlb_pitching.csv` (130,546 rows), `japanese_mlb_batting.csv` (67,187 rows), `players.csv`, `season_summary.csv` (StatsAPI season and sabermetrics stats per player-season-role)
- **DOI:** `10.34740/kaggle/dsv/10697439`
- **Build:** generated in GitHub Actions by [`japanese-mlb-players-statcast/build.py`](japanese-mlb-players-statcast/build.py) (rebuilt September 2026: player list from MLB StatsAPI, regular season only, pitch counts equal StatsAPI's; the previous version had two wrong player ids and missing seasons)
- **Article:** [Zenn](https://zenn.dev/yasumorishima/articles/kaggle-dataset-japanese-mlb-statcast)

### 2. [MLB Bat Tracking (2024-2025)](https://www.kaggle.com/datasets/yasunorim/mlb-bat-tracking-2024-2025) 🥈

MLB bat tracking leaderboard data from Baseball Savant (2024-2025).

- **Data:** 452 batters (226 per season)
- **Columns:** 19 swing metrics (bat speed, squared-up rate, blasts, swords, etc.)
- **Size:** 46 KB
- **DOI:** `10.34740/kaggle/dsv/10699103`
- **Article:** [Zenn](https://zenn.dev/yasumorishima/articles/mlb-bat-tracking-dataset)

### 3. [MLB Pitcher Arsenal (2020-2026)](https://www.kaggle.com/datasets/yasunorim/mlb-pitcher-arsenal-2020-2025)

Every pitcher's pitch mix, velocity, spin, break, whiff rate and run value by season (Baseball Savant, regular season).

- **Files:** `pitcher_arsenal.csv` (one row per pitcher, season and pitch type), `pitcher_arsenal_wide.csv` (5,716 pitcher-seasons x 129 columns), `arsenal_changes.csv` (year over year)
- **DOI:** `10.34740/kaggle/dsv/10704532`
- **Build:** generated in GitHub Actions by [`pitcher-arsenal-dataset/build.py`](pitcher-arsenal-dataset/build.py) (rebuilt September 2026: whiff rate now equals Savant's; the previous version counted called strikes in its denominator)
- **Article:** [Zenn](https://zenn.dev/yasumorishima/articles/mlb-pitcher-arsenal-dataset-2020-2025)

### 4. [MLB Statcast + Bat Tracking (2024-2025)](https://www.kaggle.com/datasets/yasunorim/mlb-statcast-bat-tracking-2024-2025)

Pitch-by-pitch Statcast data merged with Bat Tracking metrics.

- **Data:** ~1.4M rows, ~2.4 GB
- **Metrics:** bat speed, swing length, swing path tilt, Statcast pitch-by-pitch data
- **Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasumorishima/kaggle-datasets/blob/main/dataset4_statcast_bat_tracking/generate.ipynb)

### 5. [Baseball Savant Leaderboards (2024-2026)](https://www.kaggle.com/datasets/yasunorim/baseball-savant-leaderboards-2024)

20 Baseball Savant leaderboards, 2024-2026, as clean CSVs joinable by player id.

- **Files:** 20 CSVs: batting, pitching, fielding, catching, baserunning, park factors
- **Sources:** savant-extras 0.6.0 plus Savant's own CSV endpoints (Outs Above Average, Outfield Jump)
- **Build:** generated in GitHub Actions by [`savant-extras-dataset/build.py`](savant-extras-dataset/build.py) (rebuilt September 2026 for 2024-2026: the previous version had 7 tables where 2024 and 2025 were the same table)
- **Notebooks:** [Showcase](https://www.kaggle.com/code/yasunorim/savant-extras-showcase) · [Defense & Pitching Quality](https://www.kaggle.com/code/yasunorim/savant-extras-defense-pitching-quality)

### 6. [WBC 2026 Scouting - Statcast Data](https://www.kaggle.com/datasets/yasunorim/wbc-2026-scouting) 🥈

Pitch-by-pitch Statcast data for WBC 2026 roster players, 20 countries.

- **Batters:** 338,811 pitches, 18 countries
- **Pitchers:** 220,385 pitches, 14 countries
- **Summary:** 109 batters / 90 pitchers (per-player stats)
- **Rosters:** 308 MLB-affiliated players across 20 countries
- **Auto-update:** GitHub Actions ([`update-wbc-dataset.yml`](.github/workflows/update-wbc-dataset.yml)) — triggers on `workflow_dispatch`

### 7. [ABS Challenges: Triple-A 2025 to MLB 2026](https://www.kaggle.com/datasets/yasunorim/mlb-abs-challenges-aaa-2025-to-mlb-2026)

Every ABS (Automated Ball-Strike) challenge board on Baseball Savant, keyed by MLBAM player id.

- **Files:** `abs_challenges_players.csv` (12,130 rows x 211 columns; 15 boards: MLB 2025 spring test, MLB 2026 spring and regular season, Triple-A 2025 and 2026; batters, pitchers, catchers) and `abs_aaa2025_to_mlb2026.csv` (476 rows x 176 columns; batters and catchers on both the Triple-A 2025 and MLB 2026 regular-season boards, side by side)
- **Player columns:** every row also carries numbers for its own board's level, season and game type: bio from MLB StatsAPI, season hitting / pitching / catching (`api_*`), and Statcast search aggregates (`sc_*`: chase and zone swing/contact rates, xwOBA, exit velocity, catcher zone calls on takes), added by [`enrich.py`](abs-challenges-dataset/enrich.py)
- **Build:** generated in GitHub Actions by [`abs-challenges-dataset/build.py`](abs-challenges-dataset/build.py) with savant-extras 0.6.0; writes nothing unless every gate passes (the gates are listed in the dataset description)
- **Notebook:** [ABS Challenges: Who Wins Them, AAA to MLB](https://www.kaggle.com/code/yasunorim/abs-challenges-who-wins-them-aaa-to-mlb) ([source](abs-challenges-notebook/abs-challenges-starter.py))
- **DOI:** [10.34740/kaggle/dsv/20077454](https://doi.org/10.34740/kaggle/dsv/20077454)

## Workflow

### WBC 2026 Scouting (automated)

Actions → `Update WBC 2026 Kaggle Dataset` → **Run workflow**

Internally: checkout `wbc-scouting` → run `generate.py` → clean `rosters.csv` → `kaggle datasets version`

Required secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`

### Other datasets (manual)

1. Run `generate.ipynb` in Google Colab to generate CSV
2. Download CSV to local dataset folder
3. `kaggle datasets create -p <folder>` (first time) or `kaggle datasets version -p <folder> -m "message"` (update)

## Related

- [Kaggle Notebooks](https://github.com/yasumorishima/kaggle-competitions)
- [MLB Statcast Visualization](https://github.com/yasumorishima/mlb-statcast-visualization)
