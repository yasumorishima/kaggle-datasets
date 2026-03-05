# Kaggle Datasets

Baseball-themed datasets published on Kaggle, generated with [pybaseball](https://github.com/jldbc/pybaseball) and Baseball Savant.

## Published Datasets (3)

### 1. [Japanese MLB Players Statcast (2015-2025)](https://www.kaggle.com/datasets/yasunorim/japan-mlb-pitchers-batters-statcast)

Comprehensive Statcast data for 34 Japanese MLB players across 10+ seasons.

- **Pitchers:** 25 players, 118,226 pitches (2015-2025)
- **Batters:** 10 players, 56,362 batted balls (2015-2025)
- **Columns:** 238 metrics (pitch velocities, spin rates, exit velocities, launch angles, xwOBA, etc.)
- **Size:** 75.5 MB
- **DOI:** `10.34740/kaggle/dsv/10697439`
- **Article:** [Zenn](https://zenn.dev/yasumorishima/articles/kaggle-dataset-japanese-mlb-statcast)

### 2. [MLB Bat Tracking (2024-2025)](https://www.kaggle.com/datasets/yasunorim/mlb-bat-tracking-2024-2025) 🥉

MLB bat tracking leaderboard data from Baseball Savant (2024-2025).

- **Data:** 452 batters (226 per season)
- **Columns:** 19 swing metrics (bat speed, squared-up rate, blasts, swords, etc.)
- **Size:** 46 KB
- **DOI:** `10.34740/kaggle/dsv/10699103`
- **Article:** [Zenn](https://zenn.dev/yasumorishima/articles/mlb-bat-tracking-dataset)

### 3. [MLB Pitcher Arsenal Evolution (2020-2025)](https://www.kaggle.com/datasets/yasunorim/mlb-pitcher-arsenal-2020-2025)

Seasonal arsenal composition and performance metrics for MLB pitchers (2020-2025).

- **Data:** 4,253 pitcher-seasons (2020-2025)
- **Columns:** 111 metrics (pitch type usage %, velocity, movement, whiff rate, xwOBA, etc.)
- **Size:** 1.14 MB
- **DOI:** `10.34740/kaggle/dsv/10704532`
- **Article:** [Zenn](https://zenn.dev/yasumorishima/articles/mlb-pitcher-arsenal-dataset-2020-2025)

### 4. [Baseball Savant Leaderboards (2024-2025)](https://www.kaggle.com/datasets/yasunorim/baseball-savant-leaderboards-2024)

20 Baseball Savant & FanGraphs leaderboards as clean CSV files.

- **Files:** 20 CSVs — batting, pitching, fielding, catching, baserunning, park factors
- **Sources:** savant-extras v0.4.3 (17 leaderboards) + pybaseball (OAA, Outfield Jump, Pitcher Quality)
- **Notebooks:** [Showcase](https://www.kaggle.com/code/yasunorim/savant-extras-showcase) · [Defense & Pitching Quality](https://www.kaggle.com/code/yasunorim/savant-extras-defense-pitching-quality)

### 5. [WBC 2026 Scouting - Statcast Data](https://www.kaggle.com/datasets/yasunorim/wbc-2026-scouting)

Pitch-by-pitch Statcast data for WBC 2026 roster players, 20 countries.

- **Batters:** 338,811 pitches, 18 countries
- **Pitchers:** 220,385 pitches, 14 countries
- **Summary:** 109 batters / 90 pitchers (per-player stats)
- **Rosters:** 308 MLB-affiliated players across 20 countries
- **Auto-update:** GitHub Actions ([`update-wbc-dataset.yml`](.github/workflows/update-wbc-dataset.yml)) — triggers on `workflow_dispatch`

## In Progress (1)

### 5. MLB Statcast + Bat Tracking (2024-2025)

Pitch-by-pitch Statcast data with Bat Tracking metrics (bat speed, swing length, swing path tilt).

- **Status:** 🚧 Generating dataset (~2.4GB, ~1.4M rows)
- **Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasumorishima/kaggle-datasets/blob/main/dataset4_statcast_bat_tracking/generate.ipynb)

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
