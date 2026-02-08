# Dataset 3 メタデータ（Kaggle Web UI手動入力用）

## Subtitle
Track pitcher arsenal composition changes across 2020-2025 MLB seasons

## Tags
- baseball
- mlb
- sports
- statcast
- time series analysis
- data visualization

## Description

### Overview

This dataset tracks **MLB pitcher arsenal evolution** from 2020 to 2025, providing a comprehensive view of how pitchers' pitch mix and performance metrics change over time. Each row represents a pitcher-season combination with pitch type metrics in wide format.

### Data Source

- **Source**: MLB Advanced Media (Statcast via pybaseball)
- **Collection Method**: Automated data pipeline using pybaseball library
- **Update Frequency**: Seasonal (updated after each MLB season)

### Content

- **Time Period**: 2020-2025 seasons (6 seasons)
- **Rows**: 4,253 pitcher-season combinations
- **Columns**: 111 columns (3 identifiers + 18 pitch types × 6 metrics)
- **Filter**: Only pitchers with 100+ pitches per season

### Pitch Types

FF (Four-Seam Fastball), SI (Sinker), FC (Cutter), SL (Slider), CU (Curveball), CH (Changeup), FS (Splitter), KC (Knuckle Curve), FO (Forkball), EP (Eephus), KN (Knuckleball), and others

### Metrics per Pitch Type

1. **usage_pct** - Usage percentage (%)
2. **avg_speed** - Average velocity (mph)
3. **avg_spin** - Average spin rate (rpm)
4. **whiff_rate** - Swing-and-miss rate
5. **avg_pfx_x** - Average horizontal movement (inches)
6. **avg_pfx_z** - Average vertical movement (inches)

### Use Cases

- **Trend Analysis**: Track how pitchers adjust their arsenal over seasons (e.g., Kikuchi's slider revolution)
- **Injury Detection**: Identify velocity drops after injuries
- **Strategic Research**: Analyze team-level pitch mix strategies
- **Machine Learning**: Feature engineering for pitcher performance prediction
- **Visualization**: Create interactive dashboards showing arsenal evolution

### Example Analysis

See the included analysis notebook for examples:
- Individual pitcher trends (pitch mix changes over time)
- League-wide trends (which pitch types are gaining/losing popularity)
- Velocity evolution (how fastball speed changes with age)
- Correlation heatmaps (relationships between pitch metrics)

### Citation

If you use this dataset in your work, please cite:

```
MLB Pitcher Arsenal Evolution (2020-2025)
Data collected via pybaseball from MLB Advanced Media Statcast
DOI: [To be generated]
```

### Related Datasets

- [Japanese MLB Players Statcast (2015-2025)](https://www.kaggle.com/datasets/yasunorim/japan-mlb-pitchers-batters-statcast)
- [MLB Bat Tracking (2024-2025)](https://www.kaggle.com/datasets/yasunorim/mlb-bat-tracking-2024-2025)

### Acknowledgments

Data provided by MLB Advanced Media via the [pybaseball](https://github.com/jldbc/pybaseball) library.
