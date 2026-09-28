# Pitcher Arsenal: analysis notebook

This folder holds the analysis notebook for the Kaggle dataset
[MLB Pitcher Arsenal (2020-2026)](https://www.kaggle.com/datasets/yasunorim/mlb-pitcher-arsenal-2020-2025).

- `pitcher_arsenal_analysis.py` / `.ipynb`: the notebook
  [Pitcher Arsenal Analysis](https://www.kaggle.com/code/yasunorim/pitcher-arsenal-analysis)
  (league pitch mix, speed and whiff% by season; one pitcher's arsenal over time; the 2026 arsenals
  of the 20 pitchers with the most pitches; 2025 to 2026 changes).
- `kernel-metadata.json`: used by the "Push Notebook to Kaggle" workflow.

The dataset itself is built by [`pitcher-arsenal-dataset/build.py`](../pitcher-arsenal-dataset/build.py)
in GitHub Actions (rebuilt September 2026). The earlier hand-built version, its generator notebook
and its column notes were removed from this folder: its `whiff_rate` divided swinging strikes by
swings plus called strikes, and it was not limited to the regular season.
