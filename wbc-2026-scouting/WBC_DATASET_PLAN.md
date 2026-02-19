# WBC 2026 Scouting Dataset — Data Guide

Pitch-by-pitch MLB Statcast data for players selected to the **WBC 2026 (World Baseball Classic)** rosters.
All data reflects MLB regular season performance.

---

## Files

| File | Rows | Description |
|---|---|---|
| `statcast_batters.csv` | 324,099 | Pitch-by-pitch batting data, 18 countries |
| `statcast_pitchers.csv` | 217,139 | Pitch-by-pitch pitching data, 14 countries |
| `batter_summary.csv` | 105 | Per-player batting summary stats |
| `pitcher_summary.csv` | 86 | Per-player pitching summary stats |
| `rosters.csv` | 309 | Full WBC 2026 roster (20 countries) |
| `stadiums.csv` | 1,002 | MLB stadium coordinates |

---

## Key Columns

### statcast_batters.csv / statcast_pitchers.csv

| Column | Description |
|---|---|
| `country` | ISO country code (e.g. USA, JPN, DOM) |
| `country_name` | Full country name |
| `pitch_type` | Pitch type code (FF, SL, CH, CU, SI, FC, FS, ST, ...) |
| `release_speed` | Pitch velocity (mph) |
| `release_spin_rate` | Spin rate (rpm) |
| `plate_x / plate_z` | Pitch location at home plate |
| `launch_speed` | Exit velocity (mph) - batters file |
| `launch_angle` | Launch angle (degrees) - batters file |
| `estimated_woba_using_speedangle` | xwOBA |
| `bat_speed` | Bat speed (mph) - 2024+ only |
| `events` | Plate appearance outcome (single, home_run, strikeout, ...) |
| `game_date` | Date of the game |

### batter_summary.csv

`mlbam_id`, `player_name`, `country`, `PA`, `AB`, `H`, `1B`, `2B`, `3B`, `HR`, `BB`, `HBP`, `K`, `TB`, `AVG`, `OBP`, `SLG`, `OPS`, `K_pct`, `BB_pct`, `xwOBA`, `avg_exit_velo`, `avg_launch_angle`

### pitcher_summary.csv

`mlbam_id`, `player_name`, `country`, `total_pitches`, `PA_faced`, `K`, `BB`, `HR_allowed`, `H_allowed`, `opp_AVG`, `opp_SLG`, `K_pct`, `BB_pct`, `xwOBA_against`, `avg_velo`, `avg_spin_rate`, `pitch_type_count`, `primary_pitch`

### rosters.csv

`name`, `name_ja`, `country`, `pool`, `position`, `team`, `on_40_man`, `role`

---

## Country Coverage

| Code | Country | Pool | Batting | Pitching |
|---|---|---|---|---|
| USA | USA | B (Houston) | yes | yes |
| JPN | Japan | C (Tokyo) | yes | yes |
| DOM | Dominican Republic | D (Miami) | yes | yes |
| VEN | Venezuela | D (Miami) | yes | yes |
| PUR | Puerto Rico | A (San Juan) | yes | yes |
| MEX | Mexico | B (Houston) | yes | yes |
| KOR | Korea | C (Tokyo) | yes | yes |
| NED | Netherlands | D (Miami) | yes | yes |
| CAN | Canada | A (San Juan) | yes | yes |
| ITA | Italy | B (Houston) | yes | yes |
| ISR | Israel | D (Miami) | yes | yes |
| GBR | Great Britain | B (Houston) | yes | yes |
| PAN | Panama | A (San Juan) | yes | yes |
| COL | Colombia | A (San Juan) | yes | yes |
| CUB | Cuba | A (San Juan) | yes | no |
| TPE | Chinese Taipei | C (Tokyo) | yes | no |
| NCA | Nicaragua | D (Miami) | yes | no |
| AUS | Australia | C (Tokyo) | yes | no |
| BRA | Brazil | B (Houston) | no | no |
| CZE | Czechia | C (Tokyo) | no | no |

Brazil and Czechia have no MLB-affiliated players. Cuba, Chinese Taipei, Nicaragua, and Australia have no pitcher Statcast records.

---

## Quick Start

```python
import pandas as pd

# Load summary stats
bat = pd.read_csv('/kaggle/input/wbc-2026-scouting/batter_summary.csv')
pit = pd.read_csv('/kaggle/input/wbc-2026-scouting/pitcher_summary.csv')

# Top batters by xwOBA (min 100 PA)
print(bat[bat['PA'] >= 100].sort_values('xwOBA', ascending=False).head(10))

# Load raw Statcast for a specific country
usa_bat = pd.read_csv(
    '/kaggle/input/wbc-2026-scouting/statcast_batters.csv',
    low_memory=False
)
usa_bat = usa_bat[usa_bat['country'] == 'USA']
```

---

## Analysis Ideas

- Compare batting exit velocity and xwOBA across national teams
- Study pitch arsenal composition and velocity by country
- Build country-level strength ratings for WBC pool predictions
- Analyze pitch type usage trends (e.g. Sweeper adoption)
- Explore how batting approach (K%, BB%, launch angle) varies by roster

---

## Interactive Dashboards

Per-player scouting dashboards (spray charts, zone heatmaps, pitch movement) are available for all WBC teams:
**https://github.com/yasumorishima/wbc-scouting**

---

## Data Source

[Baseball Savant](https://baseballsavant.mlb.com/) via [pybaseball](https://github.com/jldbc/pybaseball).
Roster: WBC 2026 official roster (Baseball America, February 2026).
License: CC0 1.0
