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

### 2. [MLB Bat Tracking (2024-2025)](https://www.kaggle.com/datasets/yasunorim/mlb-bat-tracking-2024-2025)

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

## In Progress (1)

### 4. MLB Statcast + Bat Tracking (2024-2025)

Pitch-by-pitch Statcast data with Bat Tracking metrics (bat speed, swing length, swing path tilt).

- **Status:** 🚧 Testing data collection
- **Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasumorishima/kaggle-datasets/blob/main/dataset4_statcast_bat_tracking/generate_test.ipynb)

## Workflow

1. Run `generate.ipynb` in Google Colab to generate CSV
2. Download CSV to local dataset folder
3. `kaggle datasets create -p <folder>` (first time) or `kaggle datasets version -p <folder> -m "message"` (update)

## Related

- [Kaggle Notebooks](https://github.com/yasumorishima/kaggle-competitions)
- [MLB Statcast Visualization](https://github.com/yasumorishima/mlb-statcast-visualization)
