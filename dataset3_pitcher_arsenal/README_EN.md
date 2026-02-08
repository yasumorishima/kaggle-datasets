# MLB Pitcher Arsenal Evolution (2020-2025)

Track how MLB pitchers' pitch mix and performance metrics evolve across six seasons (2020-2025).

## 📊 Overview

This dataset provides a comprehensive view of **pitcher arsenal evolution** in Major League Baseball. Each row represents a pitcher-season combination with detailed metrics for up to 18 pitch types in wide format.

- **Time Period**: 2020-2025 seasons (6 seasons)
- **Rows**: 4,253 pitcher-season combinations
- **Columns**: 111 columns (3 identifiers + 18 pitch types × 6 metrics)
- **Filter**: Only pitchers with 100+ pitches per season
- **Format**: Wide format (one row per pitcher-season)

## 🎯 Pitch Types

FF (Four-Seam Fastball), SI (Sinker), FC (Cutter), SL (Slider), CU (Curveball), CH (Changeup), FS (Splitter), KC (Knuckle Curve), FO (Forkball), EP (Eephus), KN (Knuckleball), ST (Sweeper), SV (Slurve), and more.

## 📈 Metrics (per pitch type)

- **usage_pct**: Usage percentage (0-100%)
- **avg_speed**: Average velocity (mph)
- **avg_spin**: Average spin rate (rpm)
- **whiff_rate**: Swing-and-miss rate (0-1)
- **avg_pfx_x**: Average horizontal movement (inches)
- **avg_pfx_z**: Average vertical movement (inches, gravity-adjusted)

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yasumorishima/kaggle-datasets/blob/main/dataset3_pitcher_arsenal/pitcher_arsenal_evolution_2020_2025.ipynb)

1. Click the badge above to open in Colab
2. Run all cells in order
3. `pitcher_arsenal_evolution_2020_2025.csv` will be downloaded

**Note**: Data collection takes 30-60 minutes.

### Option 2: Local Execution

```bash
# Install required packages
pip install pybaseball duckdb pandas numpy

# Launch Jupyter Notebook
jupyter notebook pitcher_arsenal_evolution_2020_2025.ipynb
```

## 📁 Files

- `pitcher_arsenal_evolution_2020_2025.csv` - Main dataset (4,253 rows × 111 columns)
- `pitcher_arsenal_evolution_2020_2025.ipynb` - Data collection notebook
- `pitcher_arsenal_analysis.ipynb` - Example analysis notebook
- `pitcher_arsenal_cover.png` - Cover image
- `README.md` - This file

## 🎓 Use Cases

### Trend Analysis
Track how pitchers adjust their arsenal over time. For example:
- **Kikuchi's Slider Revolution**: Increased slider usage from 20% (2019) to 40+ (2022-2025)
- **Velocity Changes**: Detect post-injury velocity drops or age-related decline

### Team Strategy Research
Analyze organization-level pitch mix preferences:
- Houston Astros' heavy reliance on sliders
- Tampa Bay Rays' emphasis on four-seam fastballs

### Machine Learning Features
Use arsenal metrics as features for:
- Pitcher performance prediction (ERA, FIP, strikeout rate)
- Injury risk assessment
- Career trajectory modeling

### Data Visualization
Create interactive dashboards showing:
- Individual pitcher trends over time
- League-wide pitch type popularity shifts
- Correlation heatmaps between pitch metrics

## 📊 Data Source

- **Source**: MLB Advanced Media (Statcast)
- **Collection Method**: Automated pipeline using [pybaseball](https://github.com/jldbc/pybaseball) library
- **Update Frequency**: Seasonal (updated after each MLB season)

## 🔗 Related Datasets

- [Japanese MLB Players Statcast (2015-2025)](https://www.kaggle.com/datasets/yasunorim/japan-mlb-pitchers-batters-statcast) - Detailed pitch-by-pitch data for Japanese MLB players
- [MLB Bat Tracking (2024-2025)](https://www.kaggle.com/datasets/yasunorim/mlb-bat-tracking-2024-2025) - Bat tracking metrics for all MLB batters

## 📝 Citation

If you use this dataset in your work, please cite:

```
MLB Pitcher Arsenal Evolution (2020-2025)
Data collected via pybaseball from MLB Advanced Media Statcast
URL: https://www.kaggle.com/datasets/yasunorim/pitcher-arsenal-evolution-2020-2025
DOI: [To be generated]
```

## 🙏 Acknowledgments

Data provided by MLB Advanced Media via the [pybaseball](https://github.com/jldbc/pybaseball) library. Special thanks to the pybaseball contributors for maintaining this excellent tool.

## 📄 License

CC0: Public Domain. Data sourced from MLB Advanced Media (Statcast).
