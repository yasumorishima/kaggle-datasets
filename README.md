# Kaggle Datasets

Baseball-themed datasets published on Kaggle, generated with [pybaseball](https://github.com/jldbc/pybaseball).

## Datasets

| # | Dataset | Description | Status |
|---|---------|-------------|--------|
| 1 | [japanese-mlb-players-statcast](./japanese-mlb-players-statcast/) | Japanese MLB Players Statcast Data (2015-2025) | In Progress |
| 2 | mlb-bat-tracking | MLB Bat Tracking Leaderboard (2024-2025) | Planned |
| 3 | mlb-pitcher-arsenal-evolution | MLB Pitcher Arsenal Evolution (2020-2025) | Planned |
| 4 | mlb-statcast-2024-2025 | MLB Statcast Full Season Data (2024-2025) | Planned |

## Workflow

1. Run `generate.ipynb` in Google Colab to generate CSV
2. Download CSV to local dataset folder
3. `kaggle datasets create -p <folder>` (first time) or `kaggle datasets version -p <folder> -m "message"` (update)

## Related

- [Kaggle Notebooks](https://github.com/yasumorishima/kaggle-competitions)
- [MLB Statcast Visualization](https://github.com/yasumorishima/mlb-statcast-visualization)
